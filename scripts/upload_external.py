#!/usr/bin/env python3
"""Upload the downloaded MDC archives to the training volume.

`modal volume put` needs an explicit source and destination, and MDC's filenames
change between fetches, so this walks `data/downloads/<corpus>/` and uploads
whatever single archive it finds under a stable remote path:

    data/downloads/tts_lumajang/tts-javanese-...tar.gz
      -> uploads/external/tts_lumajang/tts-javanese-...tar.gz

`prepare_external` on the volume then resolves the corpus by directory rather
than by filename, so a renamed download cannot break the pipeline.

Nothing here names the task: the remote paths carry the corpus alias only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOLUME = "govind-llm-data"
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")

# `configs/datasets.json` and `prepare_external.SOURCES` disagree on one alias:
# downloads land in `cv_jv_spontaneous/`, and the corpus builder calls it
# `cv_jv`. Uploading under the download name would put the archive somewhere
# `resolve_external_archive` never looks, and the failure would only surface on
# the volume, after a 28 MB upload and a container start.
REMOTE_ALIAS = {"cv_jv_spontaneous": "cv_jv"}


def find_archives(downloads: Path, corpora):
    """One archive per requested corpus, or a clear error naming what is wrong."""
    found, missing = {}, []
    for corpus in corpora:
        directory = downloads / corpus
        archives = sorted(p for p in directory.iterdir()
                          if p.is_file() and p.name.endswith(ARCHIVE_SUFFIXES)
                          ) if directory.is_dir() else []
        if not archives:
            missing.append(corpus)
        elif len(archives) > 1:
            raise ValueError(f"{corpus} holds {len(archives)} archives; expected one: "
                             f"{[p.name for p in archives]}")
        else:
            found[corpus] = archives[0]
    return found, missing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("corpora", nargs="*", default=None,
                   help="Corpus aliases; default is every directory under --downloads")
    p.add_argument("--downloads", type=Path, default=ROOT / "data/downloads")
    p.add_argument("--volume", default=DEFAULT_VOLUME)
    p.add_argument("--remote-prefix", default="uploads/external")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not args.downloads.is_dir():
        print(f"error: no download directory at {args.downloads}", file=sys.stderr)
        return 2
    corpora = args.corpora or sorted(d.name for d in args.downloads.iterdir() if d.is_dir())
    if not corpora:
        print(f"error: {args.downloads} holds no corpus directories", file=sys.stderr)
        return 2
    try:
        found, missing = find_archives(args.downloads, corpora)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for corpus in missing:
        print(f"warning: no archive for {corpus}; skipping", file=sys.stderr)
    if not found:
        print("error: nothing to upload", file=sys.stderr)
        return 2

    for corpus, path in found.items():
        alias = REMOTE_ALIAS.get(corpus, corpus)
        remote = f"{args.remote_prefix}/{alias}/{path.name}"
        size = path.stat().st_size / (1024 * 1024)
        command = ["modal", "volume", "put", args.volume, str(path), remote]
        print(f"{corpus:<20} {size:8.1f} MiB -> {remote}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"error: upload failed for {corpus}", file=sys.stderr)
            return result.returncode
    print(f"\n{len(found)} archive(s) uploaded to volume {args.volume}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
