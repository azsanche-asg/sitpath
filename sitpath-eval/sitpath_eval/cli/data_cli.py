from __future__ import annotations

import argparse
import sys

from sitpath_eval.data import ETHUCYDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SitPath data utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download dataset assets.")
    download.add_argument("--dataset", default="eth_ucy", choices=["eth_ucy"])
    download.add_argument("--root", required=True, help="Target directory for dataset files.")
    return parser


def handle_download(args: argparse.Namespace) -> None:
    if args.dataset == "eth_ucy":
        ETHUCYDataset.download(args.root)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        handle_download(args)
    else:
        parser.error("Unsupported command")


if __name__ == "__main__":
    main(sys.argv[1:])
