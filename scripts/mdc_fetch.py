#!/usr/bin/env python3
"""Browse MDC and securely download/verify a dataset or deterministic subset."""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "datasets.json"
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".webm"}
METADATA_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".md", ".yaml", ".yml"}


class MDCError(RuntimeError):
    """A user-actionable MDC API or data error."""


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dataset(target: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    configured = config.get("datasets", {})
    if target in configured:
        result = dict(configured[target])
        result["alias"] = target
        return result
    for alias, dataset in configured.items():
        if dataset.get("id") == target:
            result = dict(dataset)
            result["alias"] = alias
            return result
    return {"id": target, "alias": target}


def request_json(
    url: str,
    method: str = "GET",
    token: Optional[str] = None,
    data: Optional[bytes] = None,
) -> Dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "mdc-indonesian-javanese/1.0"}
    if token:
        headers["Authorization"] = "Bearer {}".format(token)
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            detail = payload.get("error") or payload.get("message") or str(payload)
        except Exception:
            detail = exc.reason
        if exc.code == 401:
            detail = "authentication failed; create an MDC API key and export MDC_API_KEY"
        elif exc.code == 403:
            detail = "access denied; accept the dataset terms/request access in the MDC web UI"
        raise MDCError("MDC API returned HTTP {}: {}".format(exc.code, detail)) from exc
    except urllib.error.URLError as exc:
        raise MDCError("MDC API request failed: {}".format(exc.reason)) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_size: Optional[int] = None) -> None:
    """Download a presigned URL without printing it; resume a partial file when possible."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "mdc-indonesian-javanese/1.0"}
    if offset:
        headers["Range"] = "bytes={}-".format(offset)
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and expected_size and offset == expected_size:
            partial.replace(destination)
            return
        raise MDCError("Dataset storage returned HTTP {}".format(exc.code)) from exc

    status = getattr(response, "status", response.getcode())
    mode = "ab" if offset and status == 206 else "wb"
    if mode == "wb":
        offset = 0
    downloaded = offset
    next_report = ((downloaded // (64 * 1024 * 1024)) + 1) * (64 * 1024 * 1024)
    with response, partial.open(mode) as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                print("Downloaded {:.1f} MiB".format(downloaded / (1024 * 1024)), file=sys.stderr)
                next_report += 64 * 1024 * 1024

    if expected_size is not None and downloaded != expected_size:
        raise MDCError("Download size is {} bytes; expected {}".format(downloaded, expected_size))
    partial.replace(destination)


def _safe_member_path(destination: Path, member_name: str) -> Path:
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts:
        raise MDCError("Unsafe archive path: {}".format(member_name))
    return destination.joinpath(*member.parts)


def _extract_regular_file(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    target = _safe_member_path(destination, member.name)
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        return
    if not member.isfile():
        return
    source = archive.extractfile(member)
    if source is None:
        raise MDCError("Could not read archive member {}".format(member.name))
    target.parent.mkdir(parents=True, exist_ok=True)
    with source, target.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def extract_subset(
    archive_path: Path,
    destination: Path,
    max_audio_files: Optional[int],
    seed: int,
    dataset: Mapping[str, Any],
) -> Path:
    if destination.exists() and any(destination.iterdir()):
        raise MDCError("Extraction directory is not empty: {}".format(destination))
    destination.mkdir(parents=True, exist_ok=True)

    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except tarfile.TarError as exc:
        raise MDCError("Unsupported or invalid tar archive: {}".format(archive_path)) from exc

    with archive:
        members = archive.getmembers()
        audio = sorted(
            [m for m in members if m.isfile() and PurePosixPath(m.name).suffix.lower() in AUDIO_SUFFIXES],
            key=lambda member: member.name,
        )
        if not audio:
            raise MDCError("Archive contains no recognized audio files")
        if max_audio_files is not None:
            if max_audio_files < 1:
                raise MDCError("--max-audio-files must be positive")
            count = min(max_audio_files, len(audio))
            selected_audio = sorted(random.Random(seed).sample(audio, count), key=lambda member: member.name)
        else:
            selected_audio = audio

        selected_names = {member.name for member in selected_audio}
        selected = []
        for member in members:
            suffix = PurePosixPath(member.name).suffix.lower()
            is_metadata = member.isfile() and (
                suffix in METADATA_SUFFIXES
                or PurePosixPath(member.name).name.lower().startswith(("readme", "license", "copying"))
            )
            if member.isdir() or member.name in selected_names or is_metadata:
                selected.append(member)
        for member in selected:
            _extract_regular_file(archive, member, destination)

    manifest = {
        "dataset_id": dataset.get("id"),
        "dataset_alias": dataset.get("alias"),
        "dataset_name": dataset.get("name"),
        "source_archive": archive_path.name,
        "source_sha256": sha256_file(archive_path),
        "seed": seed,
        "audio_files_in_archive": len(audio),
        "audio_files_extracted": len(selected_audio),
        "selected_audio": [member.name for member in selected_audio],
        "note": "Metadata is copied whole; filter rows to the selected_audio list before training.",
    }
    manifest_path = destination / "SUBSET_MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest_path


def command_catalog(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    params: Dict[str, Any] = {"limit": args.limit}
    if args.query:
        params["q"] = args.query
    url = "{}/datasets?{}".format(config["api_base"], urllib.parse.urlencode(params))
    payload = request_json(url)
    items = payload.get("items", [])
    if args.task:
        items = [item for item in items if item.get("task", "").upper() == args.task.upper()]
    print("id\ttask\tlocale\tlicense\tsize_bytes\tname")
    for item in items:
        print(
            "{}\t{}\t{}\t{}\t{}\t{}".format(
                item.get("id", ""),
                item.get("task", ""),
                item.get("locale", ""),
                item.get("licenseAbbreviation", ""),
                item.get("sizeBytes", ""),
                str(item.get("name", "")).replace("\t", " "),
            )
        )
    print("# API matches: {}; displayed: {}".format(payload.get("total", len(items)), len(items)))
    return 0


def command_details(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    dataset = resolve_dataset(args.dataset, config)
    payload = request_json("{}/datasets/{}".format(config["api_base"], dataset["id"]))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_download(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    dataset = resolve_dataset(args.dataset, config)
    token = os.environ.get("MDC_API_KEY")
    if not token:
        raise MDCError(
            "MDC_API_KEY is not set. Create a key at "
            "https://mozilladatacollective.com/profile/credentials and export it in this shell."
        )

    details = request_json("{}/datasets/{}".format(config["api_base"], dataset["id"]))
    dataset.update(details)
    endpoint = "sample-download" if args.sample else "download"
    session = request_json(
        "{}/datasets/{}/{}".format(config["api_base"], dataset["id"], endpoint),
        method="POST",
        token=token,
        data=b"",
    )
    filename = session.get("filename") or details.get("filename") or "{}.tar.gz".format(dataset["alias"])
    output_dir = args.download_dir or ROOT / "data" / "downloads" / dataset["alias"]
    archive_path = output_dir / filename
    expected_size = int(session["sizeBytes"]) if session.get("sizeBytes") is not None else None

    expected_hash = None if args.sample else (details.get("checksum") or dataset.get("sha256"))
    if archive_path.exists() and expected_hash and sha256_file(archive_path) == expected_hash:
        print("Using verified cached archive: {}".format(archive_path))
    else:
        download_file(session["downloadUrl"], archive_path, expected_size)
        if expected_hash:
            actual_hash = sha256_file(archive_path)
            if actual_hash.lower() != str(expected_hash).lower():
                raise MDCError("SHA-256 mismatch for {}".format(archive_path))
        print("Downloaded and verified: {}".format(archive_path))

    if args.extract:
        subset_label = "all" if args.max_audio_files is None else "audio-{}-seed-{}".format(
            args.max_audio_files, args.seed
        )
        extract_dir = args.extract_dir or ROOT / "data" / "subsets" / dataset["alias"] / subset_label
        manifest = extract_subset(archive_path, extract_dir, args.max_audio_files, args.seed, dataset)
        print("Extracted subset: {}".format(extract_dir))
        print("Manifest: {}".format(manifest))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="Search the public dataset catalog")
    catalog.add_argument("--query")
    catalog.add_argument("--task", help="Filter returned rows locally, for example ASR")
    catalog.add_argument("--limit", type=int, default=100, choices=range(1, 101), metavar="1..100")
    catalog.set_defaults(func=command_catalog)

    details = subparsers.add_parser("details", help="Get public details for an alias or dataset ID")
    details.add_argument("dataset")
    details.set_defaults(func=command_details)

    download = subparsers.add_parser("download", help="Create an authenticated download session")
    download.add_argument("dataset", help="Alias from configs/datasets.json or an MDC dataset ID")
    download.add_argument("--sample", action="store_true", help="Fetch the publisher's sample instead of full data")
    download.add_argument("--download-dir", type=Path)
    download.add_argument("--extract", action="store_true")
    download.add_argument("--extract-dir", type=Path)
    download.add_argument("--max-audio-files", type=int)
    download.add_argument("--seed", type=int, default=20260905)
    download.set_defaults(func=command_download)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args, load_config())
    except (MDCError, OSError, json.JSONDecodeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

