#!/usr/bin/env python3
"""Descriptive relationship audit; no transcript strings in output.

Uses existing complete waveform measurements. No significance claims: clips
are clustered within four development speakers and two conversations.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_data import ROOT, canonical_inputs, describe, digest_file, read_tsv, strip_accents, text_key, timestamp, write_json

FEATURES = ["decoded_duration_seconds", "words", "tokens_per_second", "rms_dbfs",
            "peak_dbfs", "silence_ratio", "estimated_snr_db", "clipped_sample_ratio",
            "effective_bandwidth_hz", "spectral_centroid_hz", "spectral_flatness",
            "accent_word_fraction", "hyphen_word_fraction", "uppercase_word_fraction"]
FLAGS = {"clipping": lambda r: r["clipped_sample_ratio"] > 1e-4,
         "quiet": lambda r: r["rms_dbfs"] < -35,
         "silence": lambda r: r["silence_ratio"] > .5,
         "low_energy_contrast": lambda r: r["estimated_snr_db"] < 15,
         "token_rate": lambda r: r["tokens_per_second"] < .5 or r["tokens_per_second"] > 6}


def rank(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="stable")
    _, starts, counts = np.unique(a[order], return_index=True, return_counts=True)
    result = np.empty(len(a))
    result[order] = np.repeat(starts + (counts - 1) / 2 + 1, counts)
    return result


def correlate(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x - x.mean(), y - y.mean()
    scale = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / scale) if scale > 1e-12 else None


def group_residual(x, groups):
    x = np.asarray(x, dtype=float).copy()
    groups = np.asarray(groups)
    for g in set(groups):
        mask = groups == g
        x[mask] -= x[mask].mean()
    return x


def correlations(rows, group):
    ranks = {k: rank([r[k] for r in rows]) for k in FEATURES}
    residuals = {k: group_residual(v, [r[group] for r in rows]) for k, v in ranks.items()}
    pairs = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            pairs.append({"a": a, "b": b, "spearman": correlate(ranks[a], ranks[b]),
                          "group_adjusted_rank_correlation": correlate(residuals[a], residuals[b])})
    return pairs


def distribution_shift(train, dev):
    """Signed Cliff delta P(dev>train)-P(dev<train), plus descriptive KS D."""
    a, b = np.sort(np.asarray(train)), np.sort(np.asarray(dev))
    grid = np.unique(np.concatenate([a, b]))
    ks = np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a) -
                       np.searchsorted(b, grid, side="right") / len(b)))
    below = np.searchsorted(a, b, side="left").sum()
    above = (len(a) - np.searchsorted(a, b, side="right")).sum()
    low, high = np.percentile(a, [1, 99])
    return {"ks_d": float(ks), "cliff_delta_dev_minus_train": float((below - above) / (len(a) * len(b))),
            "train_median": float(np.median(a)), "dev_median": float(np.median(b)),
            "dev_fraction_outside_train_p01_p99": float(np.mean((b < low) | (b > high)))}


def eta_squared(values, groups):
    a = np.asarray(values, dtype=float)
    total = ((a - a.mean()) ** 2).sum()
    if total < 1e-12:
        return None
    residual = group_residual(a, groups)
    return float(1 - (residual ** 2).sum() / total)


def design_matrix(rows, columns):
    terms = [np.ones(len(rows))]
    for key in columns:
        levels = sorted({str(r[key]) for r in rows})
        terms.extend(np.array([str(r[key]) == value for r in rows], float) for value in levels[1:])
    return np.column_stack(terms)


def partial_r_squared(rows, feature, base, added):
    y = np.array([r[feature] for r in rows])
    x0, x1 = design_matrix(rows, base), design_matrix(rows, base + added)
    e0 = y - x0 @ np.linalg.lstsq(x0, y, rcond=None)[0]
    e1 = y - x1 @ np.linalg.lstsq(x1, y, rcond=None)[0]
    s0, s1 = e0 @ e0, e1 @ e1
    return float(max(0, 1 - s1 / s0)) if s0 > 1e-12 else None


def categorical(rows, a, b):
    aa, bb = sorted({str(r[a]) for r in rows}), sorted({str(r[b]) for r in rows})
    counts = np.array([[sum(str(r[a]) == x and str(r[b]) == y for r in rows) for y in bb] for x in aa])
    expected = counts.sum(axis=1, keepdims=True) @ counts.sum(axis=0, keepdims=True) / counts.sum()
    chi = np.sum((counts - expected) ** 2 / expected)
    minimum = min(len(aa) - 1, len(bb) - 1)
    v = np.sqrt(chi / (len(rows) * minimum)) if minimum else None
    return {"row_labels": aa, "column_labels": bb, "counts": counts.tolist(),
            "cramers_v": float(v) if v is not None else None}


def js_divergence(a, b):
    keys = set(a) | set(b)
    sa, sb = sum(a.values()), sum(b.values())
    if not sa or not sb:
        return None
    result = 0.0
    for key in keys:
        p, q = a.get(key, 0) / sa, b.get(key, 0) / sb
        m = (p + q) / 2
        if p:
            result += .5 * p * np.log2(p / m)
        if q:
            result += .5 * q * np.log2(q / m)
    return float(result)


def tokens(rows, transform=lambda w: w):
    return Counter(transform(w) for r in rows for w in r["tokens"])


def vocabulary_coverage(reference, target):
    output = {}
    for mode, transform in [("scoring", lambda w: w), ("casefold", str.casefold),
                             ("accent_casefold", lambda w: strip_accents(w).casefold())]:
        a, b = tokens(reference, transform), tokens(target, transform)
        output[mode] = {"token_oov_fraction": sum(n for w, n in b.items() if w not in a) / sum(b.values()),
                        "type_oov_fraction": sum(w not in a for w in b) / len(b),
                        "jensen_shannon_divergence_bits": js_divergence(a, b)}
    return output


def summarize_group(rows):
    return {"rows": len(rows), "hours": sum(r["decoded_duration_seconds"] for r in rows) / 3600,
            "words": sum(r["words"] for r in rows),
            "metrics": {k: {"mean": float(np.mean([r[k] for r in rows])),
                             "median": float(np.median([r[k] for r in rows]))} for k in FEATURES},
            "quality_flags": {k: sum(fn(r) for r in rows) for k, fn in FLAGS.items()}}


def grouped(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[str(r[key])].append(r)
    return dict(sorted(out.items()))


def session_analysis(rows):
    groups = grouped(rows, "session")
    summaries = {g: summarize_group(rs) for g, rs in groups.items()}
    all_tokens = tokens(rows)
    for g, rs in groups.items():
        own = tokens(rs)
        summaries[g]["leave_session_out_token_oov_fraction"] = sum(n for w, n in own.items() if all_tokens[w] == n) / sum(own.values())
        summaries[g]["sample_rate_hz"] = rs[0]["sample_rate_hz"]
    quality = {}
    for flag, fn in FLAGS.items():
        counts = [(g, sum(fn(r) for r in rs)) for g, rs in groups.items()]
        ranked = sorted(counts, key=lambda x: (-x[1], x[0]))
        top = {g for g, _ in ranked[:5]}
        total = sum(n for _, n in counts)
        quality[flag] = {"total_flagged_rows": total, "sessions_affected": sum(n > 0 for _, n in counts),
                         "top5_sessions": ranked[:5],
                         "top5_share_of_flags": sum(n for _, n in ranked[:5]) / total if total else 0,
                         "top5_share_of_training_rows": sum(len(groups[g]) for g in top) / len(rows)}
    word_shares = np.array([s["words"] for s in summaries.values()], float)
    word_shares /= word_shares.sum()
    return summaries, {"quality_concentration": quality,
                       "inverse_word_concentration": float(1 / np.sum(word_shares ** 2)),
                       "top10_sessions_share_of_words": float(np.sort(word_shares)[-10:].sum()),
                       "leave_session_out_token_oov_distribution": describe([s["leave_session_out_token_oov_fraction"] for s in summaries.values()])}


def length_bin(seconds):
    for low, high in [(0, 2), (2, 6), (6, 12), (12, 20), (20, 30), (30, float("inf"))]:
        if low <= seconds < high:
            return f"{low}-{high}"
    raise ValueError("Invalid duration")


def same_speaker_contrast(rows, feature):
    common = []
    for speaker, rs in grouped(rows, "speaker").items():
        ind, mixed = [r for r in rs if r["language"] == "ind"], [r for r in rs if r["language"] == "javind"]
        if ind and mixed:
            a, b = np.mean([r[feature] for r in ind]), np.mean([r[feature] for r in mixed])
            common.append({"speaker": speaker, "ind_rows": len(ind), "mixed_rows": len(mixed),
                           "mean_difference_ind_minus_mixed": float(a - b)})
    raw_a = [r[feature] for r in rows if r["language"] == "ind"]
    raw_b = [r[feature] for r in rows if r["language"] == "javind"]
    return {"raw_mean_difference_ind_minus_mixed": float(np.mean(raw_a) - np.mean(raw_b)),
            "same_speaker_difference_weighted_by_ind_counts": sum(x["ind_rows"] * x["mean_difference_ind_minus_mixed"] for x in common) / sum(x["ind_rows"] for x in common),
            "within_speaker": common}


def main():
    reports = ROOT / "reports"
    rows, _, _ = canonical_inputs(ROOT / "data/subsets/dev/all", ROOT / "data/subsets/jember/all")
    measured = json.loads((reports / "private/clip_metrics.json").read_text())
    _, raw_train = read_tsv(next((ROOT / "data/subsets/jember/all").rglob("*.tsv")))
    files = json.loads((reports / "private/source_files.json").read_text())
    enriched = {}
    for dataset, raw in rows.items():
        index = {r["id"]: r for r in measured[dataset]}
        source = {r["filename"]: r for r in files[dataset]}
        enriched[dataset] = []
        for row in raw:
            if row["id"] not in index:
                continue
            ts = text_key(row["text"]).split()
            r = {**row, **index[row["id"]], "tokens": ts}
            r["accent_word_fraction"] = sum(strip_accents(w) != w for w in ts) / len(ts)
            r["hyphen_word_fraction"] = sum("-" in w for w in ts) / len(ts)
            r["uppercase_word_fraction"] = sum(any(c.isupper() for c in w) for w in ts) / len(ts)
            r["duration_bin"] = length_bin(r["decoded_duration_seconds"])
            sf = source[Path(row["source"]).name]
            r["source_bitrate_bps"] = sf["stream_bit_rate_bps"]
            if dataset == "train":
                r["relative_session_position"] = r["start"] / sf["decoded_duration_seconds"]
            enriched[dataset].append(r)
    train, dev = enriched["train"], enriched["dev"]
    groups = {}
    for key in ["language", "speaker", "conversation"]:
        groups[key] = {g: {**summarize_group(rs), "training_vocabulary_coverage": vocabulary_coverage(train, rs)}
                       for g, rs in grouped(dev, key).items()}
    design = design_matrix(dev, ["speaker", "conversation", "language"])
    effects = {k: {"eta_squared_by_speaker": eta_squared([r[k] for r in dev], [r["speaker"] for r in dev]),
                   "eta_squared_by_conversation": eta_squared([r[k] for r in dev], [r["conversation"] for r in dev]),
                   "eta_squared_by_language": eta_squared([r[k] for r in dev], [r["language"] for r in dev]),
                   "language_partial_r_squared_after_speaker": partial_r_squared(dev, k, ["speaker"], ["language"])} for k in FEATURES}
    session_private, sessions = session_analysis(train)
    shifts = {k: distribution_shift([r[k] for r in train], [r[k] for r in dev]) for k in FEATURES}
    bins = {ds: {b: summarize_group(rs) for b, rs in grouped(enriched[ds], "duration_bin").items()}
            for ds in enriched}
    shared_bins = set(bins["dev"]) & set(bins["train"])
    total_overlap = sum(bins["dev"][b]["rows"] for b in shared_bins)
    matched_shifts = {}
    for k in FEATURES:
        weighted = sum(bins["dev"][b]["rows"] *
                       (bins["dev"][b]["metrics"][k]["mean"] - bins["train"][b]["metrics"][k]["mean"])
                       for b in shared_bins) / total_overlap
        matched_shifts[k] = float(weighted)
    folds = {}
    for convo, target in grouped(dev, "conversation").items():
        other = [r for r in dev if r["conversation"] != convo]
        folds[convo] = {"validation_rows": len(target), "validation_words": sum(r["words"] for r in target),
                        "training_dev_rows": len(other), "jember_only": vocabulary_coverage(train, target),
                        "other_conversation_only": vocabulary_coverage(other, target),
                        "jember_plus_other_conversation": vocabulary_coverage(train + other, target)}
    temporal = {}
    position_rank = rank([r["relative_session_position"] for r in train])
    sessions_id = [r["session"] for r in train]
    for k in FEATURES:
        ranked = rank([r[k] for r in train])
        temporal[k] = {"spearman": correlate(position_rank, ranked),
                       "session_adjusted_rank_correlation": correlate(group_residual(position_rank, sessions_id), group_residual(ranked, sessions_id))}
    concentration = {}
    for key in ["speaker", "conversation"]:
        counts = np.array([g["words"] for g in groups[key].values()], float)
        shares = counts / counts.sum()
        concentration[key] = {"groups": len(counts), "largest_word_share": float(shares.max()),
                              "inverse_word_concentration": float(1 / (shares @ shares))}
    filtering = {}
    for flag, predicate in {**FLAGS, "any_quality_flag": lambda r: any(fn(r) for fn in FLAGS.values())}.items():
        kept = [r for r in train if not predicate(r)]
        removed = [r for r in train if predicate(r)]
        full_vocab, kept_vocab, dev_vocab = tokens(train), tokens(kept), tokens(dev)
        lost = set(full_vocab) - set(kept_vocab)
        filtering[flag] = {"removed_rows": len(removed), "removed_row_share": len(removed) / len(train),
                           "removed_word_share": sum(r["words"] for r in removed) / sum(r["words"] for r in train),
                           "lost_training_word_types": len(lost),
                           "dev_word_occurrences_losing_training_coverage": sum(dev_vocab[w] for w in lost),
                           "retained_training_coverage": vocabulary_coverage(kept, dev)}
    boundaries = json.loads((reports / "private/timestamp_anomalies.json").read_text())["bounds"]
    boundary_ids = {r["id"] for r in boundaries}
    bounded_train = [r for r in train if r["id"] not in boundary_ids]
    boundary_sensitivity = {k: distribution_shift([r[k] for r in bounded_train], [r[k] for r in dev]) for k in FEATURES}
    output = {"schema_version": 1,
              "input_sha256": {p: digest_file(reports / p) for p in ["private/clip_metrics.json", "private/source_files.json"]},
              "methodology": {"inference": "Descriptive associations, not causal effects. No IID clip-level p-values or confidence intervals.",
                              "adjusted_ranks": "Pearson correlation of pooled average ranks after subtracting each speaker/session's mean rank.",
                              "effect_size": "Eta squared is observed variance accounted for by group means. Partial R2 adds language to speaker fixed effects.",
                              "concentration": "Inverse sum of squared group word shares; concentration measure, not an effective independent sample size.",
                              "confounding": "Speaker is nested in conversation; their independent effects are not identifiable here.",
                              "vocabulary": "OOV and Jensen-Shannon divergence are reference-text properties; no model inference or WER was measured.",
                              "ratios": "Word rate divides word count by duration; correlations involving these are partly mathematical.",
                              "training_rows": "6638 positive timestamp rows, including five measured to available audio bounds; defect flags remain in the primary audit.",
                              "feature_limits": "Clip-averaged descriptors; no native-listener validation, word alignment, language-ID, source device, or speaker embedding model."},
              "development_groups": groups,
              "categorical_associations": {f"{a}__{b}": categorical(dev, a, b) for a, b in [("speaker", "conversation"), ("speaker", "language"), ("conversation", "language")]},
              "confounded_design": {"columns": int(design.shape[1]), "rank": int(np.linalg.matrix_rank(design))},
              "development_variance_by_group": effects,
              "development_language_contrasts": {k: same_speaker_contrast(dev, k) for k in FEATURES},
              "correlations": {"dev": correlations(dev, "speaker"), "train": correlations(train, "session")},
              "train_dev_distribution_shift": shifts,
              "duration_groups": bins,
              "duration_standardized_shift": {"bins": sorted(shared_bins), "represented_dev_rows": total_overlap,
                                               "excluded_dev_rows": len(dev) - total_overlap,
                                               "mean_dev_minus_train": matched_shifts,
                                               "limitation": "Coarse post-stratification on duration bins with shared support; not causal adjustment."},
              "training_sample_rate_groups": {g: summarize_group(rs) for g, rs in grouped(train, "sample_rate_hz").items()},
              "training_sample_rate_effects": {k: eta_squared([r[k] for r in train], [r["sample_rate_hz"] for r in train]) for k in FEATURES},
              "training_session_effects": {k: eta_squared([r[k] for r in train], sessions_id) for k in FEATURES},
              "training_sessions": sessions, "training_temporal_relationships": temporal,
              "timestamp_resolution": {"rows": len(raw_train), "rows_with_integer_second_endpoints": sum(timestamp(r["start"]).is_integer() and timestamp(r["end"]).is_integer() for r in raw_train)},
              "quality_filtering_scenarios": filtering,
              "shift_excluding_boundary_defects": boundary_sensitivity,
              "development_holdout_coverage": folds, "development_word_concentration": concentration}
    write_json(reports / "relationships.json", output)
    write_json(reports / "private/session_relationships.json", session_private)
    # Numerical plotting table, no raw reference text.
    numeric = {ds: [{k: r[k] for k in FEATURES + (["speaker", "conversation", "language"] if ds == "dev" else ["session", "sample_rate_hz"])}
                    for r in rs] for ds, rs in enriched.items()}
    write_json(reports / "private/relationship_plot_data.json", numeric)
    print(json.dumps({"report": str(reports / "relationships.json"),
                      "development_rows": len(dev), "training_rows": len(train),
                      "feature_pairs_per_dataset": len(output["correlations"]["dev"]),
                      "design": output["confounded_design"]}, indent=2))


if __name__ == "__main__":
    main()
