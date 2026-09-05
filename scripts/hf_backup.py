#!/usr/bin/env python3
"""Back up source, artifacts, and reusable weights to a private HF repo."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdc_asr.hf_backup import (  # noqa: E402
    DEFAULT_REPO_NAME,
    default_local_items,
    parse_item_specs,
    upload_items,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_NAME)
    parser.add_argument("--repo-type", default="model", choices=("model", "dataset"))
    parser.add_argument("--public", action="store_true", help="Create as public if it does not exist")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--include-base-models", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true", help="Skip local artifact ZIPs")
    parser.add_argument("--path", action="append", default=[], help="Extra backup item as LOCAL=REMOTE")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        print("error: set {} in the environment".format(args.token_env), file=sys.stderr)
        return 2
    try:
        items = default_local_items(
            ROOT,
            include_base_models=args.include_base_models,
            include_artifacts=not args.no_artifacts,
        )
        items.extend(parse_item_specs(args.path))
        result = upload_items(
            token=token,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=not args.public,
            items=items,
            commit_prefix="local backup",
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
