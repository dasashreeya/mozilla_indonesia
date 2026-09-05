#!/usr/bin/env python3
"""Download a pinned public Hugging Face model and verify published LFS hashes."""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "models.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "mdc-asr/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download(url: str, path: Path, size: Optional[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "mdc-asr/1.0"}
    if offset:
        headers["Range"] = "bytes={}-".format(offset)
    request = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(request, timeout=120)
    status = getattr(response, "status", response.getcode())
    mode = "ab" if offset and status == 206 else "wb"
    if mode == "wb":
        offset = 0
    received = offset
    with response, partial.open(mode) as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            received += len(chunk)
            print("\r{}: {:.1f} MiB".format(path.name, received / (1024 * 1024)), end="", file=sys.stderr)
    print(file=sys.stderr)
    if size is not None and received != size:
        raise RuntimeError("{} has {} bytes; expected {}".format(path, received, size))
    partial.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(config["models"]))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Resolve the revision and list files without downloading")
    args = parser.parse_args(argv)

    model = config["models"][args.model]
    repo_id = model["repo_id"]
    revision = model["revision"]
    output = args.output_dir or ROOT / "models" / args.model
    api_url = "https://huggingface.co/api/models/{}?revision={}&blobs=true".format(
        repo_id, urllib.parse.quote(revision)
    )
    metadata = get_json(api_url)
    if metadata.get("sha") != revision:
        raise RuntimeError("Resolved revision {} instead of pinned {}".format(metadata.get("sha"), revision))
    siblings = {entry["rfilename"]: entry for entry in metadata.get("siblings", [])}

    if args.dry_run:
        total = 0
        for filename in model["files"]:
            if filename not in siblings:
                raise RuntimeError("Pinned model no longer lists {}".format(filename))
            entry = siblings[filename]
            lfs = entry.get("lfs") or {}
            size = lfs.get("size") or entry.get("size") or 0
            total += size
            print("{}\t{}\t{}".format(filename, size or "unknown", lfs.get("sha256", "-")))
        print("# Known total: {} bytes ({:.2f} GiB)".format(total, total / (1024 ** 3)))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    for filename in model["files"]:
        if filename not in siblings:
            raise RuntimeError("Pinned model no longer lists {}".format(filename))
        entry = siblings[filename]
        lfs = entry.get("lfs") or {}
        expected_hash = lfs.get("sha256")
        expected_size = lfs.get("size") or entry.get("size")
        destination = output / filename
        if destination.exists() and (not expected_hash or sha256_file(destination) == expected_hash):
            print("Verified cached {}".format(destination))
            continue
        encoded_filename = "/".join(urllib.parse.quote(part) for part in filename.split("/"))
        url = "https://huggingface.co/{}/resolve/{}/{}?download=true".format(
            repo_id, revision, encoded_filename
        )
        download(url, destination, expected_size)
        if expected_hash and sha256_file(destination) != expected_hash:
            raise RuntimeError("SHA-256 mismatch for {}".format(destination))
        print("Verified {}".format(destination))

    receipt = {
        "repo_id": repo_id,
        "revision": revision,
        "license": model["license"],
        "files": model["files"],
    }
    with (output / "MODEL_RECEIPT.json").open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print("Model ready: {}".format(output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, urllib.error.URLError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
