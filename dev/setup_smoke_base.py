"""Bootstrap a minimal base checkpoint + tokenizer so the SFT pipeline can
run end-to-end on a fresh box without going through full pretrain.

Pure smoke-test scaffolding: the checkpoint is random-initialized and the
tokenizer is trained on identity_conversations.jsonl alone (~2.3MB), so
gradient/activation patterns from a run against this base are NOT
physically meaningful. Use only to validate the monitor + on-disk pipeline
at H100 scale.

Outputs (under $NANOCHAT_BASE_DIR, default ~/.cache/nanochat):
  identity_conversations.jsonl
  tokenizer/tokenizer.pkl
  tokenizer/token_bytes.pt
  base_checkpoints/<tag>/model_000000.pt
  base_checkpoints/<tag>/meta_000000.json

Run:
  python -m dev.setup_smoke_base --tag=d6_smoke --depth=6
"""

import argparse
import json
import os
import sys
import urllib.request

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataclasses import asdict
from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

IDENTITY_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"


def download_identity_jsonl(base_dir: str) -> str:
    target = os.path.join(base_dir, "identity_conversations.jsonl")
    if os.path.exists(target):
        print(f"[ok] identity_conversations.jsonl already at {target}")
        return target
    os.makedirs(base_dir, exist_ok=True)
    print(f"downloading {IDENTITY_URL} -> {target}")
    urllib.request.urlretrieve(IDENTITY_URL, target)
    print(f"[ok] saved {os.path.getsize(target):,} bytes")
    return target


def train_smoke_tokenizer(jsonl_path: str, base_dir: str, vocab_size: int) -> RustBPETokenizer:
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    if os.path.exists(pkl_path) and os.path.exists(bytes_path):
        print(f"[ok] tokenizer already at {tokenizer_dir}")
        return RustBPETokenizer.from_directory(tokenizer_dir)

    def text_iter():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                msgs = json.loads(line)
                for m in msgs:
                    content = m.get("content", "")
                    if isinstance(content, str) and content:
                        yield content

    print(f"training rustbpe tokenizer (vocab_size={vocab_size}) on {jsonl_path}")
    tok = RustBPETokenizer.train_from_iterator(text_iter(), vocab_size)
    tok.save(tokenizer_dir)

    # write token_bytes.pt (replicates scripts/tok_train.py logic)
    vsz = tok.get_vocab_size()
    special_set = set(tok.get_special_tokens())
    token_bytes = []
    for tid in range(vsz):
        s = tok.decode([tid])
        if s in special_set:
            token_bytes.append(0)
        else:
            token_bytes.append(len(s.encode("utf-8")))
    tb = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    with open(bytes_path, "wb") as f:
        torch.save(tb, f)
    print(f"[ok] token_bytes.pt saved ({vsz} entries)")
    return tok


def save_random_base(base_dir: str, tag: str, depth: int, vocab_size: int,
                     n_embd: int, n_head: int, n_kv_head: int, sequence_len: int) -> str:
    ckpt_dir = os.path.join(base_dir, "base_checkpoints", tag)
    model_path = os.path.join(ckpt_dir, "model_000000.pt")
    meta_path = os.path.join(ckpt_dir, "meta_000000.json")
    if os.path.exists(model_path) and os.path.exists(meta_path):
        print(f"[ok] base checkpoint already at {ckpt_dir}")
        return ckpt_dir

    cfg = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        window_pattern="L",
        use_smear=False,
        use_resid_lambdas=False,
        use_value_residual=False,
        use_backout=False,
    )
    print(f"building random-init GPT with cfg={cfg}")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[ok] saved model state_dict to {model_path}")

    meta = {
        "model_config": asdict(cfg),
        "user_config": {
            # leave hyperparams unset so SFT inherits sane defaults via fallback
        },
        "step": 0,
        "note": "random-init smoke base produced by dev/setup_smoke_base.py",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ok] saved meta to {meta_path}")
    return ckpt_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="d6_smoke")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--vocab-size", type=int, default=2048,
                   help="small vocab is fine for a smoke test")
    p.add_argument("--n-embd", type=int, default=384,
                   help="kept small to keep param count low")
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-kv-head", type=int, default=6)
    p.add_argument("--sequence-len", type=int, default=2048)
    args = p.parse_args()

    base_dir = get_base_dir()
    os.makedirs(base_dir, exist_ok=True)
    print(f"NANOCHAT_BASE_DIR = {base_dir}")

    jsonl_path = download_identity_jsonl(base_dir)
    train_smoke_tokenizer(jsonl_path, base_dir, args.vocab_size)
    save_random_base(
        base_dir, args.tag, args.depth, args.vocab_size,
        args.n_embd, args.n_head, args.n_kv_head, args.sequence_len,
    )
    print("done.")


if __name__ == "__main__":
    main()
