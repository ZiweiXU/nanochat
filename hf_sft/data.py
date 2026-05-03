"""Data pipeline for HF SFT with per-token role mask.

Builds (input_ids, attention_mask, labels, role_mask) tensors from a list
of message-list conversations. Each row is one conversation, right-padded
to `max_seq_len`. Conversations longer than `max_seq_len` are dropped with
a warning (truncating the prompt would corrupt the role boundary; truncating
the answer destroys the SFT target).

Role mask values (also exposed in grad_monitor_hf.PAD/PROMPT/ANSWER):
    0 = PAD     attention_mask == 0  (right-pad tail)
    1 = PROMPT  attention_mask == 1 AND labels == -100
    2 = ANSWER  labels != -100

Boundary detection uses **differential tokenization** of the chat template:
for each assistant turn, render `messages[:i]` with `add_generation_prompt=True`
to find where the assistant content begins, and `messages[:i+1]` with
`add_generation_prompt=False` to find where it ends. This works with any
HF tokenizer that has a chat template, without relying on
`return_assistant_tokens_mask` (which requires `{% generation %}` markers
in the template).
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import torch
from torch.utils.data import Dataset


PAD, PROMPT, ANSWER = 0, 1, 2


@dataclass
class SFTRow:
    input_ids: torch.Tensor       # (T,) int64
    attention_mask: torch.Tensor  # (T,) int64
    labels: torch.Tensor          # (T,) int64
    role_mask: torch.Tensor       # (T,) uint8


def _render_and_count(tokenizer, messages, add_generation_prompt: bool) -> int:
    """Tokenize the chat-template-rendered prefix and return its length."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
    return len(enc["input_ids"])


def encode_conversation(
    tokenizer,
    messages: List[dict],
    max_seq_len: int,
    pad_token_id: Optional[int] = None,
) -> Optional[SFTRow]:
    """Render one conversation, return SFTRow or None if it doesn't fit.

    Assistant tokens in any number of assistant turns are marked ANSWER;
    everything else (system, user, special markers, BOS) is PROMPT; right-pad
    is PAD.
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id  # last-resort fallback
        if pad_token_id is None:
            raise ValueError("tokenizer has no pad_token_id or eos_token_id")

    # Full rendering (tokens for the whole conversation).
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors=None)["input_ids"]
    if len(full_ids) > max_seq_len:
        return None

    # Walk turns, marking assistant ranges via differential tokenization.
    role_mask_list = [PROMPT] * len(full_ids)
    labels_list = [-100] * len(full_ids)

    last_end = 0
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        # Render up to (but not including) this assistant message, with the
        # assistant generation prompt appended. The prefix ends right where
        # this assistant message's content begins.
        n_before = _render_and_count(tokenizer, messages[:i], add_generation_prompt=True)
        # Render up to and including this assistant message.
        n_after = _render_and_count(tokenizer, messages[: i + 1], add_generation_prompt=False)

        # Defensive: if the chat template injects post-assistant boilerplate
        # (e.g. trailing <|im_end|>\n) it is included in n_after but we still
        # want it marked ANSWER so the loss covers it (model learns to emit
        # the end-of-turn token). That matches conventional SFT behavior.
        n_before = max(n_before, last_end)
        n_after = min(n_after, len(full_ids))
        if n_after <= n_before:
            continue
        for k in range(n_before, n_after):
            role_mask_list[k] = ANSWER
            labels_list[k] = full_ids[k]
        last_end = n_after

    # Pad to max_seq_len.
    pad_len = max_seq_len - len(full_ids)
    input_ids = full_ids + [pad_token_id] * pad_len
    attn = [1] * len(full_ids) + [0] * pad_len
    labels = labels_list + [-100] * pad_len
    role = role_mask_list + [PAD] * pad_len

    return SFTRow(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attn, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        role_mask=torch.tensor(role, dtype=torch.uint8),
    )


def collate_sft(rows: List[SFTRow]) -> dict:
    """Stack a list of SFTRow into a model-ready batch."""
    return {
        "input_ids":      torch.stack([r.input_ids for r in rows], dim=0),
        "attention_mask": torch.stack([r.attention_mask for r in rows], dim=0),
        "labels":         torch.stack([r.labels for r in rows], dim=0),
        "role_mask":      torch.stack([r.role_mask for r in rows], dim=0),
    }


# -----------------------------------------------------------------------------
# Conversation sources

def load_identity_jsonl(path: str) -> List[List[dict]]:
    """Load nanochat's identity_conversations.jsonl (one JSON list per line)."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)
            assert isinstance(msgs, list), f"expected list of messages, got {type(msgs)}"
            out.append(msgs)
    return out


def load_smoltalk(split: str = "train") -> List[List[dict]]:
    """Load HuggingFaceTB/smoltalk's `messages` field (already in the right shape)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split=split)
    return [row["messages"] for row in ds]


_MMLU_LETTERS = ("A", "B", "C", "D")


def _render_mc(question: str, choices: List[str]) -> str:
    """Match nanochat's `tasks.common.render_mc` so the prompt-format
    distribution is identical between the two SFT paths. Letter is
    placed AFTER the choice (better small-model binding) and there is
    no whitespace between '=' and the letter (different token id)."""
    q = f"Multiple Choice question: {question}\n"
    q += "".join(f"- {choice}={letter}\n" for letter, choice in zip(_MMLU_LETTERS, choices))
    q += "\nRespond only with the letter of the correct answer."
    return q


def load_mmlu(split: str = "auxiliary_train", limit: Optional[int] = None) -> List[List[dict]]:
    """Load cais/mmlu and convert each row to a [user, assistant] message
    pair. Mirrors `tasks.mmlu.MMLU.get_example`. The assistant message is
    the single answer letter — short ANSWER spans, but matches what the
    nanochat path trains on."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split=split).shuffle(seed=42)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    out: List[List[dict]] = []
    for row in ds:
        choices = row["choices"]
        if len(choices) != 4:
            continue
        user_msg = _render_mc(row["question"], choices)
        assistant_msg = _MMLU_LETTERS[row["answer"]]
        out.append([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ])
    return out


_GSM_TOOL_RE = re.compile(r"<<[^>]+>>")


def _render_gsm_solution(answer: str) -> str:
    """Strip GSM8K's `<<expr=result>>` calculator markers, leaving the
    plain reasoning + final `#### N` line. The nanochat path keeps the
    markers as tool calls, but for HF chat templates we need string
    content; the `#### N` final-answer marker is preserved so eval
    extractors keep working."""
    return _GSM_TOOL_RE.sub("", answer)


def load_gsm8k(subset: str = "main", split: str = "train",
               limit: Optional[int] = None) -> List[List[dict]]:
    """Load openai/gsm8k as [user, assistant] message pairs. Calculator
    tool markers are stripped — the model sees natural reasoning text
    ending in `#### N`."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    out: List[List[dict]] = []
    for row in ds:
        out.append([
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": _render_gsm_solution(row["answer"])},
        ])
    return out


# -----------------------------------------------------------------------------
# Tokenized-row dataset

class SFTDataset(Dataset):
    """Eager-tokenized SFT dataset: encodes every conversation up front and
    drops over-length rows. Suitable for tens-of-thousands of conversations
    (no streaming yet — keep this simple for the first iteration)."""

    def __init__(
        self,
        conversations: Iterable[List[dict]],
        tokenizer,
        max_seq_len: int,
        seed: int = 42,
        verbose: bool = True,
    ):
        rows: List[SFTRow] = []
        n_total = 0
        n_dropped = 0
        for msgs in conversations:
            n_total += 1
            row = encode_conversation(tokenizer, msgs, max_seq_len=max_seq_len)
            if row is None:
                n_dropped += 1
                continue
            rows.append(row)
        if verbose:
            print(f"[SFTDataset] kept {len(rows):,} / {n_total:,} conversations "
                  f"({n_dropped:,} too long for T={max_seq_len})")
        self._rows = rows
        random.Random(seed).shuffle(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx) -> SFTRow:
        return self._rows[idx]


def make_default_dataset(tokenizer, max_seq_len: int, *,
                         identity_path: Optional[str] = None,
                         use_smoltalk: bool = False,
                         smoltalk_split: str = "train",
                         use_mmlu: bool = False,
                         mmlu_split: str = "auxiliary_train",
                         mmlu_limit: Optional[int] = None,
                         use_gsm8k: bool = False,
                         gsm8k_split: str = "train",
                         gsm8k_limit: Optional[int] = None) -> SFTDataset:
    """Pragmatic default dataset: identity_conversations + any subset of
    {SmolTalk, MMLU auxiliary_train, GSM8K train}. Limits exist because
    MMLU auxiliary_train is ~100k rows — full mixture overwhelms the
    SmolTalk + identity scale by 30×, distorting the role-mask stats."""
    convs: List[List[dict]] = []
    if identity_path is not None and os.path.exists(identity_path):
        convs.extend(load_identity_jsonl(identity_path))
    if use_smoltalk:
        convs.extend(load_smoltalk(split=smoltalk_split))
    if use_mmlu:
        convs.extend(load_mmlu(split=mmlu_split, limit=mmlu_limit))
    if use_gsm8k:
        convs.extend(load_gsm8k(split=gsm8k_split, limit=gsm8k_limit))
    if not convs:
        raise ValueError("make_default_dataset needs at least one source "
                         "(identity_path or use_smoltalk/mmlu/gsm8k)")
    return SFTDataset(convs, tokenizer=tokenizer, max_seq_len=max_seq_len)
