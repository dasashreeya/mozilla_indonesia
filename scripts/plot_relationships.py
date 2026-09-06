#!/usr/bin/env python3
"""Render shareable aggregate relationship figures with Matplotlib."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, TEAL = "#2368A2", "#D97728", "#208777"


def main():
    report = json.loads((ROOT / "reports/relationships.json").read_text())
    data = json.loads((ROOT / "reports/private/relationship_plot_data.json").read_text())
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12,
                         "savefig.facecolor": "white"})
    fig, axs = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    fig.suptitle("Indonesian–Javanese: relationships that matter for validation", fontsize=19)
    ax = axs[0, 0]
    for ds, label, color in [("train", "Jember utterances (6,638)", BLUE), ("dev", "Development clips (372)", ORANGE)]:
        x = np.sort([r["decoded_duration_seconds"] for r in data[ds]])
        ax.step(x, np.arange(1, len(x) + 1) / len(x) * 100, where="post", label=label, color=color, lw=2)
    ax.set(title="1  The largest shift is clip length", xlabel="Decoded duration (seconds)", ylabel="Clips at or below duration (%)", xlim=(0, 40), ylim=(0, 101))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=.18)

    ax = axs[0, 1]
    ids = ["1", "6", "5", "7"]
    mixed = [sum(r["speaker"] == s and r["language"] == "javind" for r in data["dev"]) for s in ids]
    ind = [sum(r["speaker"] == s and r["language"] == "ind" for r in data["dev"]) for s in ids]
    ax.bar(ids, mixed, label="Mixed language", color=BLUE)
    ax.bar(ids, ind, bottom=mixed, label="Indonesian only", color=ORANGE)
    ax.set_xticks(range(4), ["Speaker 1\nConvo 5", "Speaker 6\nConvo 5", "Speaker 5\nConvo 2", "Speaker 7\nConvo 2"])
    for i, (m, n) in enumerate(zip(mixed, ind)):
        ax.text(i, m + n + 3, str(m + n), ha="center", fontsize=10)
    ax.set(title="2  Language coverage depends on speaker", ylabel="Development clips", ylim=(0, 190))
    ax.legend(fontsize=9, loc="upper right")

    ax = axs[0, 2]
    keys = ["rms_dbfs", "estimated_snr_db"]
    contrasts = report["development_language_contrasts"]
    raw = [contrasts[k]["raw_mean_difference_ind_minus_mixed"] for k in keys]
    adjusted = [contrasts[k]["same_speaker_difference_weighted_by_ind_counts"] for k in keys]
    x = np.arange(2)
    ax.bar(x - .18, raw, .36, label="All clips", color=ORANGE)
    ax.bar(x + .18, adjusted, .36, label="Same speakers (5 and 7)", color=TEAL)
    ax.axhline(0, color="#555", lw=.8)
    ax.set_xticks(x, ["RMS level", "Frame-energy contrast*"])
    ax.set(title="3  Apparent language differences shrink", ylabel="Mean difference: Indonesian − mixed (dB)")
    ax.legend(fontsize=9, loc="best")
    ax.text(.01, -.18, "*Energy contrast is not calibrated acoustic SNR.", transform=ax.transAxes, fontsize=9)

    ax = axs[1, 0]
    keys = ["rms_dbfs", "spectral_centroid_hz", "estimated_snr_db", "silence_ratio", "effective_bandwidth_hz"]
    labels = ["RMS level", "Spectral centroid", "Energy contrast", "Relative silence", "Effective bandwidth"]
    values = [100 * report["development_variance_by_group"][k]["eta_squared_by_speaker"] for k in keys]
    ax.barh(labels[::-1], values[::-1], color=BLUE)
    for i, value in enumerate(values[::-1]):
        ax.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=10)
    ax.set(title="4  Acoustic variation clusters by speaker", xlabel="Observed variance accounted for by speaker means (%)", xlim=(0, 100))

    ax = axs[1, 1]
    folds = report["development_holdout_coverage"]
    ids = ["2", "5"]
    a = [100 * folds[c]["jember_only"]["scoring"]["token_oov_fraction"] for c in ids]
    b = [100 * folds[c]["jember_plus_other_conversation"]["scoring"]["token_oov_fraction"] for c in ids]
    x = np.arange(2)
    ax.bar(x - .18, a, .36, label="Jember vocabulary", color=BLUE)
    ax.bar(x + .18, b, .36, label="Jember + other conversation", color=TEAL)
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - .18, va + .3, f"{va:.1f}%", ha="center", fontsize=10)
        ax.text(i + .18, vb + .3, f"{vb:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x, ["Hold out conversation 2", "Hold out conversation 5"])
    ax.set(title="5  Validation directions differ lexically", ylabel="Reference word occurrences unseen in training (%)", ylim=(0, max(a + b) + 8))
    ax.legend(fontsize=9, loc="upper right")

    ax = axs[1, 2]
    quality = report["training_sessions"]["quality_concentration"]
    keys = ["clipping", "quiet", "silence"]
    a = [100 * quality[k]["top5_share_of_flags"] for k in keys]
    b = [100 * quality[k]["top5_share_of_training_rows"] for k in keys]
    x = np.arange(3)
    ax.bar(x - .18, a, .36, label="Share of flagged rows", color=ORANGE)
    ax.bar(x + .18, b, .36, label="Share of all training rows", color=BLUE)
    ax.set_xticks(x, ["Near-full-scale", "Very quiet", "Mostly silent"])
    ax.set(title="6  Quality flags concentrate in recordings", ylabel="Share in each flag's top five sessions (%)", ylim=(0, 95))
    ax.legend(fontsize=9, loc="upper right")
    fig.savefig(out / "data_relationships.png", dpi=180)
    fig.savefig(out / "data_relationships.svg")
    plt.close(fig)

    keys = ["decoded_duration_seconds", "words", "tokens_per_second", "rms_dbfs", "silence_ratio",
            "estimated_snr_db", "effective_bandwidth_hz", "accent_word_fraction"]
    names = ["Duration", "Words", "Words/sec", "RMS", "Silence", "Energy contrast", "Bandwidth", "Accents"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), layout="constrained")
    fig.suptitle("Correlations before and after accounting for recording groups", fontsize=17)
    for i, (ds, group) in enumerate([("dev", "speaker"), ("train", "session")]):
        for j, field in enumerate(["spearman", "group_adjusted_rank_correlation"]):
            matrix = np.eye(len(keys))
            for pair in report["correlations"][ds]:
                if pair["a"] in keys and pair["b"] in keys:
                    ia, ib = keys.index(pair["a"]), keys.index(pair["b"])
                    matrix[ia, ib] = matrix[ib, ia] = pair[field] if pair[field] is not None else np.nan
            ax = axes[i, j]
            im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(len(keys)), names, rotation=40, ha="right", fontsize=9)
            ax.set_yticks(range(len(keys)), names, fontsize=9)
            ax.set_title(f"{'Development' if ds == 'dev' else 'Jember'} — {'raw ranks' if j == 0 else group + '-adjusted ranks'}")
            for y in range(len(keys)):
                for x in range(len(keys)):
                    value = matrix[y, x]
                    ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(value) > .6 else "#222")
    fig.colorbar(im, ax=axes, shrink=.6, label="Rank correlation (descriptive; not causal)")
    fig.savefig(out / "relationship_correlations.png", dpi=180)
    fig.savefig(out / "relationship_correlations.svg")
    print("Rendered two relationship figures as PNG and SVG")


if __name__ == "__main__":
    main()
