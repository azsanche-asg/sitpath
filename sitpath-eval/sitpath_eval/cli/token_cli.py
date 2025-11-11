from __future__ import annotations

import argparse
from pathlib import Path

from sitpath_eval.data import ETHUCYDataset
from sitpath_eval.tokens import SitPathTokenizer, Vocabulary, precompute_tokens

DATASETS = {
    "eth_ucy": ETHUCYDataset,
}

ARTIFACTS_DIR = Path("artifacts")
VOCAB_FILE = ARTIFACTS_DIR / "vocab.json"
TOKENS_DIR = ARTIFACTS_DIR / "tokens"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SitPath tokenization utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-vocab", help="Scan dataset and build tokenizer vocabulary.")
    build.add_argument("--dataset", choices=DATASETS.keys(), default="eth_ucy")
    build.add_argument("--split", default="train")
    build.add_argument("--root", default="./data", help="Root directory with dataset files.")
    build.add_argument("--out", default=str(VOCAB_FILE), help="Output vocab JSON path.")

    prec = subparsers.add_parser("precompute", help="Encode trajectories into cached token files.")
    prec.add_argument("--dataset", choices=DATASETS.keys(), default="eth_ucy")
    prec.add_argument("--split", default="train")
    prec.add_argument("--root", default="./data")
    prec.add_argument("--vocab", default=str(VOCAB_FILE))
    prec.add_argument("--out-dir", default=str(TOKENS_DIR))

    return parser


def handle_build_vocab(args: argparse.Namespace) -> None:
    dataset_cls = DATASETS[args.dataset]
    trajectories = dataset_cls.load_split(args.root, split=args.split)
    dataset = dataset_cls(trajectories)

    vocab = Vocabulary()
    tokenizer = SitPathTokenizer(vocab=vocab)
    for item in dataset:
        tokenizer.encode_trajectory(item["pos"].numpy())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(out_path)
    print(f"[sitpath-eval] Saved vocabulary with {len(vocab)} tokens to {out_path}")


def handle_precompute(args: argparse.Namespace) -> None:
    dataset_cls = DATASETS[args.dataset]
    trajectories = dataset_cls.load_split(args.root, split=args.split)
    dataset = dataset_cls(trajectories)

    vocab = Vocabulary.load(args.vocab)
    tokenizer = SitPathTokenizer(vocab=vocab)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.npz"
    precompute_tokens(dataset, tokenizer, str(out_path))
    print(f"[sitpath-eval] Saved token cache to {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-vocab":
        handle_build_vocab(args)
    elif args.command == "precompute":
        handle_precompute(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
