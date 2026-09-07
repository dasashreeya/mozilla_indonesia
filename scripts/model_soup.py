#!/usr/bin/env python3
"""Average several checkpoints into one model -- ensembling at zero inference cost.

Both the 1st- and 2nd-place Word Track solutions of the closest comparable
competition finished this way, and unlike a decode-time ensemble the result is a
single model, so it costs nothing against the two-hour runtime limit.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CONFIG_FILES = ("config.json", "generation_config.json", "preprocessor_config.json",
                "processor_config.json", "tokenizer_config.json", "tokenizer.json",
                "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
                "added_tokens.json", "normalizer.json")


def state_dict_files(directory: Path):
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors weights in {directory}")
    return files


def average_state_dicts(directories, output: Path, weights=None):
    """Uniformly (or `weights`-) average parameters shared by every checkpoint.

    Averaging is done in float32 and cast back, so repeated bf16 rounding does
    not accumulate across the members.
    """
    import torch
    from safetensors.torch import load_file, save_file

    if len(directories) < 2:
        raise ValueError("Averaging needs at least two checkpoints")
    if weights is None:
        weights = [1.0 / len(directories)] * len(directories)
    if len(weights) != len(directories) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must match the checkpoints and sum to 1")

    totals, dtypes, shard_of = {}, {}, {}
    for weight, directory in zip(weights, directories):
        seen = set()
        for shard in state_dict_files(directory):
            tensors = load_file(str(shard))
            for name, tensor in tensors.items():
                seen.add(name)
                shard_of.setdefault(name, shard.name)
                if name not in totals:
                    if directory is not directories[0]:
                        raise ValueError(f"{directory} adds unseen parameter {name}")
                    totals[name] = tensor.to(torch.float32) * weight
                    dtypes[name] = tensor.dtype
                else:
                    if totals[name].shape != tensor.shape:
                        raise ValueError(f"Shape mismatch for {name} in {directory}")
                    totals[name] += tensor.to(torch.float32) * weight
            del tensors
        missing = set(totals) - seen
        if missing:
            raise ValueError(f"{directory} is missing {len(missing)} parameters, "
                             f"e.g. {sorted(missing)[:3]}")

    output.mkdir(parents=True, exist_ok=True)
    shards = {}
    for name, tensor in totals.items():
        shards.setdefault(shard_of[name], {})[name] = tensor.to(dtypes[name])
    for shard_name, tensors in shards.items():
        save_file(tensors, str(output / shard_name), metadata={"format": "pt"})
    for name in CONFIG_FILES + ("model.safetensors.index.json",):
        source = directories[0] / name
        if source.is_file():
            shutil.copy2(source, output / name)
    return {"output": str(output), "members": [str(d) for d in directories],
            "weights": weights, "parameters": len(totals), "shards": sorted(shards)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, action="append", required=True,
                   help="Repeat for each member of the soup")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weight", type=float, action="append",
                   help="Optional per-checkpoint weight; defaults to uniform")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite an existing soup")
    receipt = average_state_dicts(args.checkpoint, args.output, args.weight)
    (args.output / "SOUP_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
