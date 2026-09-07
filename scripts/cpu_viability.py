"""CPU-only, grouped evaluation of saved ASR hypotheses; never launches inference.

Uses numpy and the repository's official-normalizer implementation. Reference-
assisted oracles are explicitly separate from predictions made without answers.
Run from the repository root with ``python3 scripts/cpu_viability.py``.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdc_asr.normalization import normalize_text
from mdc_asr.wer import word_errors

PIVOT = "w0-qwen-ft-greedy"
SMALL_POOL = [PIVOT, "arm-combo", "arm-kitchen", "w0-whisper-jw"]
# Fixed before looking at this experiment's scores. These are candidate sets,
# not claims that all entries are linguistically equivalent in every context.
VARIANT_GROUPS = [
    ("ning", "neng", "nang"), ("nggak", "gak"), ("kaya", "kayak"),
    ("karna", "karena"), ("iki", "ki"), ("kandhang", "kandang"),
    ("lek", "nek"),
]


def tokens(text):
    return normalize_text(text).split()


def append_token(row, token, ref):
    new = [row[0] + 1]
    for i, word in enumerate(ref, 1):
        new.append(min(row[i] + 1, new[i - 1] + 1,
                       row[i - 1] + (word != token)))
    return new


def distance(ref, hyp):
    row = list(range(len(ref) + 1))
    for token in hyp:
        row = append_token(row, token, ref)
    return row[-1]


def decompose(pivot, hyp):
    """Align into alternating gap/token slots, retaining whole insertions."""
    d = [list(range(len(hyp) + 1))]
    for i, token in enumerate(pivot, 1):
        row = [i]
        for j, other in enumerate(hyp, 1):
            row.append(min(d[i - 1][j] + 1, row[j - 1] + 1,
                           d[i - 1][j - 1] + (token != other)))
        d.append(row)
    slots = [[] for _ in range(2 * len(pivot) + 1)]
    i, j = len(pivot), len(hyp)
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (pivot[i - 1] != hyp[j - 1]):
            slots[2 * i - 1].append(hyp[j - 1])
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            slots[2 * i].append(hyp[j - 1])
            j -= 1
    result = [tuple(reversed(s)) for s in slots]
    assert [w for s in result for w in s] == hyp
    return result


def minimum_edits(ref, network):
    row = list(range(len(ref) + 1))
    for alternatives in network:
        end_rows = []
        for sequence in alternatives:
            candidate = row
            for token in sequence:
                candidate = append_token(candidate, token, ref)
            end_rows.append(candidate)
        row = [min(values) for values in zip(*end_rows)]
    return row[-1]


def vote(parts, pool, pivot_weight=1, whisper_weight=1, insertions=True):
    result = []
    for slot, keep in enumerate(parts[PIVOT]):
        if slot % 2 == 0 and not insertions:
            result.extend(keep)
            continue
        counts = Counter()
        for model in pool:
            weight = pivot_weight if model == PIVOT else (
                whisper_weight if "whisper" in model else 1)
            counts[parts[model][slot]] += weight
        chosen = max(sorted(counts), key=lambda c: (counts[c], c == keep))
        result.extend(chosen)
    return " ".join(result)


def configs():
    result = [{"name": "unchanged_pivot", "pool": [PIVOT]}]
    pools = {
        "three_qwen": SMALL_POOL[:3],
        "three_qwen_whisper": SMALL_POOL,
        "four_qwen": [PIVOT, "arm-control", "arm-combo", "arm-kitchen"],
        "language_external": [PIVOT, "arm-lang", "arm-kitchen"],
    }
    for label, pool in pools.items():
        for pw in [1, 2, 3]:
            for ww in ([0.25, 0.5, 1] if "w0-whisper-jw" in pool else [1]):
                for insertions in [False, True]:
                    result.append({"name": f"{label}:p{pw}:w{ww}:insert{int(insertions)}",
                                   "pool": pool, "pivot_weight": pw,
                                   "whisper_weight": ww, "insertions": insertions})
    return result


def group_folds(records):
    """Each group is scored once; development is held out as one conversation."""
    jav = sorted({r["group"] for r in records if r["source"] == "jember"})
    random.Random(20260907).shuffle(jav)
    folds = []
    for k in range(5):
        held = set(jav[k::5])
        train = [i for i, r in enumerate(records)
                 if r["source"] == "jember" and r["group"] not in held]
        test = [i for i, r in enumerate(records) if r["group"] in held]
        folds.append((f"jember-fold-{k}", train, test))
    train = [i for i, r in enumerate(records) if r["source"] == "jember"]
    test = [i for i, r in enumerate(records) if r["source"] == "development"]
    folds.append(("transfer-jember-to-development", train, test))
    for _, train, test in folds:
        assert not {records[i]["group"] for i in train} & {records[i]["group"] for i in test}
    assert sorted(i for _, _, test in folds for i in test) == list(range(len(records)))
    return folds


def summarize(errors, records, indices=None):
    indices = list(range(len(records))) if indices is None else list(indices)
    result = {}
    for source in ["all", "jember", "development"]:
        selected = [i for i in indices if source == "all" or records[i]["source"] == source]
        words = sum(records[i]["words"] for i in selected)
        edits = sum(int(errors[i]) for i in selected)
        result[source] = {"errors": edits, "words": words, "clips": len(selected),
                          "wer": edits / words if words else None}
    return result


def bootstrap(errors, baseline, records):
    groups = sorted({r["group"] for r in records if r["source"] == "jember"})
    counts = []
    wins = ties = 0
    for group in groups:
        ids = [i for i, r in enumerate(records) if r["group"] == group]
        delta = sum(int(errors[i] - baseline[i]) for i in ids)
        counts.append((delta, sum(records[i]["words"] for i in ids)))
        wins += delta < 0
        ties += delta == 0
    rng = random.Random(20260907)
    draws = []
    for _ in range(5000):
        sample = rng.choices(counts, k=len(counts))
        draws.append(sum(d for d, _ in sample) / sum(w for _, w in sample))
    draws.sort()
    return {"jember_session_delta_95_percentile": [draws[125], draws[4874]],
            "sessions_improved": wins, "sessions_tied": ties,
            "sessions_worsened": len(groups) - wins - ties,
            "qualification": "Conditional exploratory interval; no training-seed uncertainty or independent development conversations. Fitted folds share training groups."}


def feature_data(records, train_rows, pool=SMALL_POOL):
    """Candidate features never use evaluation reference text or session IDs.

    A small conditional softmax uses model support, KEEP, length and training-
    corpus unigram/bigram context. Its supervision uses only each fold's training
    reference alignments. Features are fixed, numerical and intentionally small.
    """
    uni, bi = Counter(), Counter()
    for row in train_rows:
        words = ["<s>"] + tokens(row["transcript"]) + ["</s>"]
        uni.update(words)
        bi.update(zip(words, words[1:]))
    vocab = len(uni)
    all_features, all_labels, owner, slot_ids, offsets, choices = [], [], [], [], [0], []
    for clip, r in enumerate(records):
        pivot = r["hyp"][PIVOT]
        # Reference-derived labels are kept separate from features and decoding.
        target_slots = decompose(pivot, r["ref"])
        clip_choices = []
        for slot, keep in enumerate(r["parts"][PIVOT]):
            alternatives = sorted({r["parts"][m][slot] for m in pool})
            clip_choices.append(alternatives)
            left_index = slot // 2 - 1
            right_index = slot // 2 + (slot % 2)
            left = pivot[left_index] if left_index >= 0 else "<s>"
            right = pivot[right_index] if right_index < len(pivot) else "</s>"
            for alternative in alternatives:
                support = [float(r["parts"][m][slot] == alternative) for m in pool]
                # Fixed feature positions across pools; absent models have no vote.
                support += [0.] * (4 - len(support))
                sequence = [left] + list(alternative) + [right]
                log_context = sum(math.log((bi[a, b] + 0.1) / (uni[a] + 0.1 * vocab))
                                  for a, b in zip(sequence, sequence[1:])) / (len(sequence) - 1)
                common = sum(math.log1p(uni[t]) for t in alternative) / max(1, len(alternative))
                row = support + [
                    float(alternative == keep), float(not alternative),
                    float(slot % 2 == 0), min(len(alternative), 5) / 5,
                    min(sum(map(len, alternative)), 30) / 30,
                    common / 10, log_context / 15,
                    float(len(alternative) == 1 and len(keep) == 1 and
                          alternative[0][:3] == keep[0][:3]),
                ]
                # Allow support/KEEP decisions to differ for insertion gaps.
                row += [x * float(slot % 2 == 0) for x in support]
                all_features.append(row)
                all_labels.append(float(alternative == target_slots[slot]))
                owner.append(clip)
                slot_ids.append(len(offsets) - 1)
            offsets.append(len(all_features))
        choices.append(clip_choices)
    return {"x": np.asarray(all_features, dtype=np.float64),
            "y": np.asarray(all_labels), "owner": np.asarray(owner),
            "slot": np.asarray(slot_ids), "offsets": np.asarray(offsets),
            "choices": choices}


def fit_selector(data, train_indices, steps=220):
    train_mask = np.isin(data["owner"], train_indices)
    has_label = np.add.reduceat(data["y"], data["offsets"][:-1]) > 0
    # Unanimous slots have no choice and contribute no gradient.
    nontrivial = np.diff(data["offsets"]) > 1
    mask = train_mask & has_label[data["slot"]] & nontrivial[data["slot"]]
    x, y = data["x"][mask], data["y"][mask]
    _, ids = np.unique(data["slot"][mask], return_inverse=True)
    starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    w = np.zeros(x.shape[1])
    w[4] = 1  # Predeclared mild preference for KEEP.
    if len(starts) == 0:
        return w
    momentum = np.zeros_like(w)
    for _ in range(steps):
        logits = x @ w
        ex = np.exp(logits - np.maximum.reduceat(logits, starts)[ids])
        probability = ex / np.add.reduceat(ex, starts)[ids]
        gradient = x.T @ (probability - y) / len(starts) + 0.002 * w
        momentum = 0.9 * momentum + gradient
        w -= 0.15 * momentum
    return w


def decode_selector(data, weights, records, indices):
    # No references/labels are consulted here.
    scores = data["x"] @ weights
    slot_number = 0
    predictions = {}
    selected = set(indices)
    for clip, r in enumerate(records):
        output = []
        for local_slot, alternatives in enumerate(data["choices"][clip]):
            start, end = data["offsets"][slot_number:slot_number + 2]
            if clip in selected:
                values = scores[start:end].copy()
                # Fixed conservative margin for changing the pivot.
                for j, candidate in enumerate(alternatives):
                    if candidate == r["parts"][PIVOT][local_slot]:
                        values[j] += 0.5
                chosen = int(np.argmax(values))
                output.append(alternatives[chosen])
            slot_number += 1
        if clip in selected:
            predictions[clip] = " ".join(t for seq in output for t in seq)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/recovery-jav1")
    parser.add_argument("--predictions", type=Path, default=ROOT / "reports/private/problem-brief-audit-20260906/recovered/preds")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/private/cpu-viability-20260906")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    receipt = json.loads((args.data / "PREPARATION_RECEIPT.json").read_text())
    for filename, key in [("metadata.csv", "metadata_sha256"),
                          ("train.qwen.jsonl", "train_jsonl_sha256"),
                          ("validation.qwen.jsonl", "validation_jsonl_sha256")]:
        hashes[filename] = hashlib.sha256((args.data / filename).read_bytes()).hexdigest()
        assert hashes[filename] == receipt[key], f"Changed preparation: {filename}"
    prov = json.loads((args.data / "private_provenance.json").read_text())
    train_rows = [r for r in prov if r["split"] == "train"]
    records = [dict(r) for r in prov if r["split"] == "validation"]
    expected = {r["audio_filename"] for r in records}
    assert len(expected) == len(records)
    systems = {}
    for path in sorted(args.predictions.glob("*.csv")):
        rows = list(csv.DictReader(path.open()))
        systems[path.stem] = {r["audio_filename"]: r["transcript"] for r in rows}
        assert len(rows) == len(expected) == len(systems[path.stem])
        assert set(systems[path.stem]) == expected
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    train_sessions = {x.split(":")[0] for r in train_rows if r["source"] == "jember" for x in r["source_rows"]}
    for r in records:
        r["ref"] = tokens(r["transcript"])
        r["words"] = len(r["ref"])
        if r["source"] == "jember":
            sessions = {x.split(":")[0] for x in r["source_rows"]}
            assert len(sessions) == 1 and not sessions & train_sessions
            r["group"] = "jember:" + next(iter(sessions))
        else:
            r["group"] = "development:" + str(r["convo_id"])
        r["hyp"] = {m: tokens(p[r["audio_filename"]]) for m, p in systems.items()}
        r["parts"] = {m: decompose(r["hyp"][PIVOT], h) for m, h in r["hyp"].items()}
    folds = group_folds(records)
    print(f"Verified {len(records)} clips, {len(systems)} prediction files and disjoint sessions.", flush=True)
    baseline = np.asarray([distance(r["ref"], r["hyp"][PIVOT]) for r in records])
    for i, r in enumerate(records):
        assert baseline[i] == word_errors(r["transcript"], systems[PIVOT][r["audio_filename"]]).errors
    configurations = configs()
    error_matrix = np.zeros((len(configurations), len(records)), dtype=int)
    predictions = []
    for c, config in enumerate(configurations):
        output = []
        for i, r in enumerate(records):
            if c == 0:
                hyp = systems[PIVOT][r["audio_filename"]]
            else:
                hyp = vote(r["parts"], **{k: v for k, v in config.items() if k != "name"})
            output.append(hyp)
            error_matrix[c, i] = distance(r["ref"], tokens(hyp))
        predictions.append(output)
        if c % 10 == 0:
            print(f"Scored voting configuration {c + 1}/{len(configurations)}.", flush=True)
    methods = {"baseline": baseline}
    outputs = {"baseline": predictions[0]}
    selected = baseline.copy()
    selected_outputs = predictions[0].copy()
    fold_receipts = []
    for label, train, test in folds:
        best = int(np.argmin(error_matrix[:, train].sum(axis=1)))
        selected[test] = error_matrix[best, test]
        for i in test:
            selected_outputs[i] = predictions[best][i]
        fold_receipts.append({"fold": label, "training_groups": sorted({records[i]["group"] for i in train}),
                              "test_groups": sorted({records[i]["group"] for i in test}),
                              "chosen_configuration": configurations[best]})
    methods["group_selected_vote"] = selected
    outputs["group_selected_vote"] = selected_outputs
    fixed = next(i for i, c in enumerate(configurations) if c["name"] == "three_qwen:p1:w1:insert1")
    methods["fixed_three_qwen_vote"] = error_matrix[fixed]
    outputs["fixed_three_qwen_vote"] = predictions[fixed]
    in_sample_best = int(np.argmin(error_matrix.sum(axis=1)))
    print("Voting complete; fitting the fixed-feature contextual selector by held-out group.", flush=True)
    learned_weights = []
    selector_pools = {
        "group_fitted_contextual_selector": SMALL_POOL,
        "contextual_three_qwen": SMALL_POOL[:3],
        "contextual_qwen_whisper": [PIVOT, "w0-whisper-jw"],
    }
    for name, pool in selector_pools.items():
        features = feature_data(records, train_rows, pool)
        contextual = baseline.copy()
        contextual_outputs = predictions[0].copy()
        for label, train, test in folds:
            weights = fit_selector(features, train)
            decoded = decode_selector(features, weights, records, test)
            for i, hyp in decoded.items():
                contextual[i] = distance(records[i]["ref"], tokens(hyp))
                contextual_outputs[i] = hyp
            learned_weights.append({"method": name, "fold": label, "weights": weights.tolist()})
        print(f"Contextual selector finished {name} ({len(pool)} ASR models).", flush=True)
        methods[name] = contextual
        outputs[name] = contextual_outputs

    variants = {w: {(x,) for x in group} for group in VARIANT_GROUPS for w in group}
    oracle_values = defaultdict(list)
    for i, r in enumerate(records):
        pivot = r["hyp"][PIVOT]
        pairs = [variants.get(w, {(w,)}) for w in pivot]
        oracle_values["listed_variants_only"].append(minimum_edits(r["ref"], pairs))
        oracle_values["whole_clip_all_nine"].append(min(distance(r["ref"], h) for h in r["hyp"].values()))
        for label, pool in [("word_three_qwen", SMALL_POOL[:3]),
                            ("word_three_qwen_whisper", SMALL_POOL),
                            ("word_all_nine", list(systems))]:
            network = [{r["parts"][m][s] for m in pool} for s in range(len(r["parts"][PIVOT]))]
            oracle_values[label].append(minimum_edits(r["ref"], network))
            if label == "word_all_nine":
                extended = []
                for alternatives in network:
                    expanded = set(alternatives)
                    for seq in alternatives:
                        if len(seq) == 1:
                            expanded.update(variants.get(seq[0], {seq}))
                    extended.append(expanded)
                oracle_values["word_all_nine_plus_listed_variants"].append(minimum_edits(r["ref"], extended))
        if i % 100 == 0:
            print(f"Oracle diagnostic {i + 1}/{len(records)}.", flush=True)

    scores = {name: summarize(e, records) for name, e in methods.items()}
    intervals = {name: bootstrap(e, baseline, records) for name, e in methods.items() if name != "baseline"}
    for name, output in outputs.items():
        with (args.output / f"{name}.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["audio_filename", "transcript"])
            writer.writerows((r["audio_filename"], output[i]) for i, r in enumerate(records))
    # Independently score final emitted strings with the official S/D/I scorer.
    for name, output in outputs.items():
        actual = [word_errors(r["transcript"], output[i]).errors for i, r in enumerate(records)]
        assert actual == methods[name].tolist(), f"Official scorer mismatch: {name}"
    words = sum(r["words"] for r in records)
    target_errors = math.floor(0.148 * words)
    without35 = [i for i, r in enumerate(records) if r["group"] != "jember:35"]
    result = {
        "protocol": "Exploratory cross-fitting on previously inspected validation data. Five Jember session folds; entire development conversation scored using a method fitted/selected on Jember only. No development references used for fitting or configuration selection.",
        "hashes": hashes, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline": PIVOT, "scores": scores, "intervals": intervals,
        "all_configurations": [{**c, "scores": summarize(error_matrix[i], records)} for i, c in enumerate(configurations)],
        "in_sample_best_vote_diagnostic_only": {"configuration": configurations[in_sample_best], "scores": summarize(error_matrix[in_sample_best], records)},
        "folds": fold_receipts, "contextual_selector_weights": learned_weights,
        "selector_pools": selector_pools,
        "oracle_diagnostics": {k: summarize(v, records) for k, v in oracle_values.items()},
        "variant_groups": VARIANT_GROUPS,
        "target_scenario": {"local_wer": 0.148, "words": words, "maximum_errors": target_errors,
                            "baseline_errors": int(sum(baseline)), "required_net_errors_removed": int(sum(baseline)) - target_errors},
        "excluding_session35_sensitivity_only": {k: summarize(v, records, without35) for k, v in methods.items()},
        "per_clip": [{"audio_filename": r["audio_filename"], "group": r["group"], "source": r["source"], "words": r["words"],
                      "errors": {k: int(v[i]) for k, v in methods.items()},
                      "oracle_errors": {k: int(v[i]) for k, v in oracle_values.items()}} for i, r in enumerate(records)],
        "seconds": time.monotonic() - started,
        "limitations": [
            "No new ASR training, audio decoding or VibeVoice runtime was tested.",
            "Local-to-public score conversion is not established.",
            "Oracle paths use references and are not deployable scores or universal bounds.",
            "The seven variant groups are a narrow diagnostic, not a complete convention model.",
            "This small selector tests one feature set and training objective, not all contextual correction methods.",
            "All source ASR predictions predate this audit; the holdout has been used for past model selection.",
            "Cross-fitted scores do not describe one single frozen trained corrector; a fresh fold is needed for confirmation.",
            "Inference costs of the multi-model pools have not been validated in the competition container.",
        ],
    }
    (args.output / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["scores", "intervals", "oracle_diagnostics", "target_scenario", "seconds"]}, indent=2))


if __name__ == "__main__":
    main()
