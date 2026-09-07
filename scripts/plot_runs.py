#!/usr/bin/env python3
"""Plot the training programme from run receipts and evaluation JSONs.

Four figures, each answering a question that decides what to run next:

* `arms.png`      -- which arm is actually best, split by language, because the
                     two halves of this task move independently and a combined
                     number hides that;
* `curves.png`    -- where the WER optimum sits inside a run, which epoch-level
                     validation could not see;
* `sessions.png`  -- per-session WER, which is how session 35's timestamp
                     misalignment was found and how the training corpus gets
                     audited for the same defect;
* `target.png`    -- every measured system on one WER scale against the target.

Also writes `summary.csv`. That table is not decoration: three of the five
series colours sit below 3:1 contrast on the chart surface, so the palette
validator requires relief -- direct labels on the marks and a table view.

Reads a directory holding `evals/*.json` (from eval_checkpoint.py) and
`runs/*/BEST_MODEL.json` (from train_recovery.py), so it works unchanged
against a local copy or a mounted Modal volume.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Validated categorical palette, light surface, adjacent pairlist. Slots are
# assigned in fixed order and never cycled; a sixth run folds into "other".
# Verified with the data-viz palette validator: worst adjacent CVD dE 9.1,
# worst adjacent normal-vision dE 19.6, contrast WARN answered by direct
# labels plus summary.csv.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e4e0"
CRITICAL = "#c8352f"
GOOD = "#1baf7a"

# Reference points this project already owns, from experiments/results.csv.
PUBLIC_LEADER = 0.2196
PUBLIC_TARGET = 0.20
LOCAL_TO_PUBLIC = 0.052


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 14,
        "xtick.color": INK_MUTED, "ytick.color": INK_SOFT,
        "xtick.labelsize": 8, "ytick.labelsize": 8.5,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.dpi": 160,
    })


def recess(ax, axis="x"):
    """Grid and spines stay behind the data, which is the point of the figure."""
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def wer(counts):
    n = counts.get("reference_words") or 0
    if not n:
        return None
    return (counts.get("substitutions", 0) + counts.get("deletions", 0)
            + counts.get("insertions", 0)) / n


def merge(parts):
    total = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0}
    for part in parts:
        for key in total:
            total[key] += part.get(key, 0)
    return total


def language_split(groups, test_jav_share):
    """Javanese WER, non-Javanese WER, and the two reweighted to the test set.

    The validation fold is 37.96% Javanese by reference words; the test set is
    about 49%. A fold-weighted mean therefore understates the half that carries
    most of the error, so both weightings are reported and the test-weighted one
    is what the figures sort on.
    """
    jav = groups.get("language:jav")
    others = [v for k, v in groups.items()
              if k.startswith("language:") and k != "language:jav"]
    jav_wer = wer(jav) if jav else None
    other_wer = wer(merge(others)) if others else None
    fold = wer(merge([g for k, g in groups.items() if k.startswith("language:")]))
    if jav_wer is None or other_wer is None:
        return jav_wer, other_wer, fold, fold
    weighted = test_jav_share * jav_wer + (1 - test_jav_share) * other_wer
    return jav_wer, other_wer, fold, weighted


def eval_paths(root):
    """Every scored result on the volume, however it was produced.

    eval_checkpoint.py writes to `evals/`, but a training run writes its own
    per-checkpoint metrics under `runs/<arm>/<label>.metrics.json`. Reading only
    the first leaves the arms off every figure -- the wave that produced them is
    exactly what the figures are for -- so both are collected, and the run files
    are labelled `<arm>/<label>` so a checkpoint can be traced to its run.
    """
    paths = sorted((root / "evals").glob("*.json"))
    for run in sorted((root / "runs").glob("*")):
        if run.is_dir():
            paths.extend(sorted(run.glob("*.metrics.json")))
    return paths


def load_evals(root):
    rows = []
    for path in eval_paths(root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"skipping unreadable eval: {path}")
            continue
        groups = data.get("groups") or {}
        jav, other, fold, weighted = language_split(groups, ARGS.test_jav_share)
        label = data.get("label") or path.stem
        if path.parent.parent.name == "runs":
            label = f"{path.parent.name}/{path.name.split('.')[0]}"
        # Run-derived labels are "<arm>/baseline", so a startswith check misses
        # them and the untrained model's ~0.50 stretches every axis.
        if label.rsplit("/", 1)[-1].startswith("baseline"):
            continue
        rows.append({"label": label, "path": path,
                     "wer": data.get("wer"), "jav": jav, "other": other,
                     "fold": fold, "weighted": weighted, "groups": groups})
    return rows


def load_curves(root):
    runs = []
    for path in sorted((root / "runs").glob("*/BEST_MODEL.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        points = [(h.get("step"), h.get("wer")) for h in data.get("epochs") or []
                  if h.get("step") is not None and h.get("wer") is not None]
        if len(points) >= 2:
            runs.append({"run": path.parent.name, "points": sorted(points),
                         "baseline": data.get("baseline_wer")})
    return runs


def pct(value, _=None):
    return f"{value:.2f}"


def plot_arms(rows, out):
    """Grouped bars: the two halves of the task, per arm, on one WER axis."""
    rows = [r for r in rows if r["jav"] is not None and r["other"] is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: r["weighted"], reverse=True)
    labels = [r["label"] for r in rows]
    height = max(3.0, 0.52 * len(rows) + 1.9)
    fig, ax = plt.subplots(figsize=(8.4, height))
    y = range(len(rows))
    bar = 0.36
    # 2px surface gap between adjacent bars, per the mark spec.
    ax.barh([i + bar / 2 + 0.012 for i in y], [r["jav"] for r in rows], height=bar,
            color=SERIES[0], label="Javanese only", zorder=3)
    ax.barh([i - bar / 2 - 0.012 for i in y], [r["other"] for r in rows], height=bar,
            color=SERIES[1], label="Code-switched", zorder=3)
    for i, r in enumerate(rows):
        ax.text(r["jav"] + 0.004, i + bar / 2 + 0.012, f'{r["jav"]:.4f}', va="center",
                fontsize=7.6, color=INK_SOFT, family="DejaVu Sans Mono")
        ax.text(r["other"] + 0.004, i - bar / 2 - 0.012, f'{r["other"]:.4f}', va="center",
                fontsize=7.6, color=INK_SOFT, family="DejaVu Sans Mono")
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("WER (lower is better)")
    ax.set_xlim(0, max(r["jav"] for r in rows) * 1.16)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_title("Every arm, split by language")
    # Outside the axes: inside, it lands on whichever bar happens to be shortest.
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.005), ncols=2)
    recess(ax)
    fig.text(0.008, 0.008, f"Sorted by WER reweighted to the test set's "
             f"{ARGS.test_jav_share:.0%} Javanese share.", fontsize=7.4, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_curves(runs, out):
    """Where the optimum sits inside a run -- the thing epoch-level eval misses."""
    if not runs:
        return None
    runs = sorted(runs, key=lambda r: r["run"])[:len(SERIES)]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, run in enumerate(runs):
        steps = [p[0] for p in run["points"]]
        wers = [p[1] for p in run["points"]]
        ax.plot(steps, wers, color=SERIES[i], linewidth=2, marker="o",
                markersize=4.5, zorder=3, label=run["run"])
        best = min(range(len(wers)), key=lambda k: wers[k])
        # Ring the optimum in the surface colour so it reads over the line, and
        # label it there rather than at the line end: runs stop at different
        # steps, so end-of-line labels land on top of each other.
        ax.plot(steps[best], wers[best], marker="o", markersize=9,
                markerfacecolor=SERIES[i], markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)
        ax.annotate(f"{wers[best]:.4f}", (steps[best], wers[best]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=7.8, color=INK_SOFT, family="DejaVu Sans Mono")
    # The zero-shot baseline is ~0.43 against runs in the 0.26-0.30 band. Drawing
    # it would spend two thirds of the axis on empty space, so it is stated in the
    # caption instead of plotted.
    baselines = [r["baseline"] for r in runs if r["baseline"] and math.isfinite(r["baseline"])]
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("held-out WER")
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_title("Validation WER inside each run")
    ax.margins(x=0.06, y=0.16)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.005), ncols=min(len(runs), 4))
    recess(ax, axis="y")
    if baselines:
        fig.text(0.008, 0.008, f"Starting point {max(baselines):.4f}, off-scale above. "
                 "The ringed point is each run's optimum.", fontsize=7.4, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sessions(rows, out, label=None):
    """Per-session WER, sorted. Outliers here are usually bad labels, not bad audio."""
    chosen = None
    for row in rows:
        if label and row["label"] != label:
            continue
        sessions = {k.split(":", 1)[1]: v for k, v in row["groups"].items()
                    if k.startswith("session:")}
        if len(sessions) >= 5:
            chosen = (row["label"], sessions)
            break
    if not chosen:
        return None
    name, sessions = chosen
    scored = sorted(((s, wer(c), c.get("reference_words", 0)) for s, c in sessions.items()),
                    key=lambda t: (t[1] is None, -(t[1] or 0)))
    scored = [s for s in scored if s[1] is not None]
    if not scored:
        return None
    values = sorted(v for _, v, _ in scored)
    median = values[len(values) // 2]
    p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
    fig, ax = plt.subplots(figsize=(8.4, max(3.2, 0.24 * len(scored) + 1.9)))
    ys = list(range(len(scored)))[::-1]
    for y, (session, value, _) in zip(ys, scored):
        outlier = value > p90
        colour = CRITICAL if outlier else SERIES[0]
        ax.plot([0, value], [y, y], color=GRID, linewidth=1.1, zorder=2)
        ax.plot([value], [y], marker="o", markersize=6, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        if outlier:
            ax.text(value + 0.012, y, f"{value:.3f}  suspect alignment", va="center",
                    fontsize=7.6, color=CRITICAL, family="DejaVu Sans Mono")
    ax.axvline(median, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"median {median:.3f}", (median, 0.99), xycoords=ax.get_xaxis_transform(),
                textcoords="offset points", xytext=(4, -9), fontsize=7.6, color=INK_MUTED)
    ax.set_yticks(ys, [s for s, _, _ in scored])
    ax.set_xlabel("WER (lower is better)")
    ax.set_xlim(0, max(v for _, v, _ in scored) * 1.32)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_title(f"Per-session WER — {name}")
    recess(ax)
    fig.text(0.008, 0.008, "Sessions past the 90th percentile are flagged: a session far "
             "outside the distribution is usually a timestamp misalignment, not hard audio.",
             fontsize=7.4, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_target(rows, out, limit=18):
    """One scale: the best systems, the leader, and the target.

    Capped: a wave leaves dozens of checkpoints, and a chart with seventy rows
    is a table with extra steps. The question this figure answers -- how close is
    the best work to the leader -- needs only the front of the field.
    """
    points = [(r["label"], r["weighted"] + LOCAL_TO_PUBLIC) for r in rows
              if r["weighted"] is not None]
    if not points:
        return None
    points.sort(key=lambda p: p[1])
    points = points[:limit]
    fig, ax = plt.subplots(figsize=(8.4, max(2.6, 0.36 * len(points) + 2.0)))
    ax.axvspan(0, PUBLIC_TARGET, color=GOOD, alpha=0.09, zorder=0)
    ax.axvline(PUBLIC_TARGET, color=GOOD, linewidth=1.4, zorder=2)
    ax.axvline(PUBLIC_LEADER, color=CRITICAL, linewidth=1.4,
               linestyle=(0, (4, 3)), zorder=2)
    ys = list(range(len(points)))[::-1]
    for y, (name, value) in zip(ys, points):
        beats = value < PUBLIC_LEADER
        ax.plot([0, value], [y, y], color=GRID, linewidth=1.1, zorder=1)
        ax.plot([value], [y], marker="o", markersize=7,
                color=GOOD if beats else SERIES[0],
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)
        ax.text(value + 0.006, y, f"{value:.4f}", va="center", fontsize=8,
                color=INK_SOFT, family="DejaVu Sans Mono")
    blend = ax.get_xaxis_transform()
    ax.annotate(f"target {PUBLIC_TARGET:.2f}", (PUBLIC_TARGET, 0.99), xycoords=blend,
                textcoords="offset points", xytext=(-5, -9), ha="right",
                fontsize=7.8, color=GOOD)
    ax.annotate(f"leader {PUBLIC_LEADER:.4f}", (PUBLIC_LEADER, 0.99), xycoords=blend,
                textcoords="offset points", xytext=(6, -9), fontsize=7.8, color=CRITICAL)
    ax.set_yticks(ys, [n for n, _ in points])
    ax.set_xlabel("forecast public WER")
    ax.set_xlim(0, max(max(v for _, v in points) * 1.18, PUBLIC_LEADER * 1.3))
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_title("Forecast against the leaderboard")
    recess(ax)
    fig.text(0.008, 0.008, f"Test-weighted local WER + {LOCAL_TO_PUBLIC:.3f}. The offset is "
             "a two-point calibration; treat these as ranking, not as scores.",
             fontsize=7.4, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary(rows, path):
    columns = ["label", "wer_fold", "wer_test_weighted", "wer_javanese",
               "wer_code_switched", "forecast_public"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["weighted"] is None, r["weighted"] or 0)):
            writer.writerow({
                "label": row["label"],
                "wer_fold": f'{row["fold"]:.6f}' if row["fold"] is not None else "",
                "wer_test_weighted": f'{row["weighted"]:.6f}' if row["weighted"] is not None else "",
                "wer_javanese": f'{row["jav"]:.6f}' if row["jav"] is not None else "",
                "wer_code_switched": f'{row["other"]:.6f}' if row["other"] is not None else "",
                "forecast_public": (f'{row["weighted"] + LOCAL_TO_PUBLIC:.6f}'
                                    if row["weighted"] is not None else ""),
            })
    return path


def main(argv=None):
    global ARGS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="Directory holding evals/ and runs/")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--test-jav-share", type=float, default=0.49,
                   help="Javanese share of the test set by reference words")
    p.add_argument("--session-label", default="",
                   help="Eval label to draw the per-session figure from; default is "
                        "the first eval carrying at least five sessions")
    ARGS = p.parse_args(argv)
    if not 0 < ARGS.test_jav_share < 1:
        p.error("test-jav-share must be a fraction")
    style()
    ARGS.output.mkdir(parents=True, exist_ok=True)
    rows = load_evals(ARGS.root)
    runs = load_curves(ARGS.root)
    written = []
    for produced in (
        plot_arms(rows, ARGS.output / "arms.png"),
        plot_curves(runs, ARGS.output / "curves.png"),
        plot_sessions(rows, ARGS.output / "sessions.png", ARGS.session_label or None),
        plot_target(rows, ARGS.output / "target.png"),
    ):
        if produced:
            written.append(str(produced))
    if rows:
        written.append(str(write_summary(rows, ARGS.output / "summary.csv")))
    result = {"evals": len(rows), "runs": len(runs), "written": written}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
