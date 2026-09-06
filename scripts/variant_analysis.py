#!/usr/bin/env python3
"""Measure how much WER is spelling-variant confusion rather than mishearing.

The fine-tuned model's top confusions are overwhelmingly alternate spellings of
the SAME word (nggak/gak/enggak, opo/apa, iso/isa, sampai/sampe). Those are a
transcription-convention problem, not an acoustic one, so a text-side map could
recover them -- but only if the references are internally consistent about which
spelling they use.

Reports, in order:
  1. how much of the substitution mass is near-identical strings,
  2. whether the references are self-consistent (a pair appearing in BOTH
     directions caps what any global rule can recover),
  3. a two-fold cross-validated estimate of the gain from a majority-vote map,
     built on one half of the clips and scored on the other.

Usage:
    PYTHONPATH=src python3 scripts/variant_analysis.py GT.csv PRED.csv
"""
import argparse
import collections
import csv
import difflib
import sys
from pathlib import Path

from mdc_asr.normalization import normalize_text
from mdc_asr.wer import ErrorCounts, word_errors


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return {r["audio_filename"]: r["transcript"] for r in csv.DictReader(fh)}


def align_subs(ref_text, hyp_text):
    """Yield (ref_word, hyp_word) for aligned substitutions only."""
    ref, hyp = normalize_text(ref_text).split(), normalize_text(hyp_text).split()
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        for k in range(min(i2 - i1, j2 - j1)):
            yield ref[i1 + k], hyp[j1 + k]


def similar(a, b, threshold=0.6):
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio() >= threshold


def corpus_wer(gt, pred, mapping=None, keys=None):
    total = ErrorCounts()
    for k in (keys if keys is not None else gt):
        hyp = pred.get(k, "")
        if mapping:
            hyp = " ".join(mapping.get(w, w) for w in normalize_text(hyp).split())
        total += word_errors(gt[k], hyp)
    return total


def reference_counts(gt, keys):
    counts = collections.Counter()
    for k in keys:
        counts.update(normalize_text(gt[k]).split())
    return counts


def build_map(gt, pred, keys, min_count=3, threshold=0.6, safe_only=False):
    """ref->hyp confusions on `keys`; map hyp spelling to the majority ref form.

    With safe_only, keep a rewrite ONLY when the hypothesis spelling never
    appears in the references at all. Both spellings of most confusions
    (gak/nggak, itu/tuh) are legitimate, so an unconditional rewrite breaks
    more correct words than it repairs -- see the cross-validated numbers.
    """
    counts = reference_counts(gt, keys) if safe_only else None
    votes = collections.defaultdict(collections.Counter)
    for k in keys:
        for r, h in align_subs(gt[k], pred.get(k, "")):
            if r != h and similar(r, h, threshold):
                votes[h][r] += 1
    mapping = {}
    for hyp_word, counter in votes.items():
        if safe_only and counts[hyp_word] > 0:
            continue
        ref_word, n = counter.most_common(1)[0]
        if n >= min_count and hyp_word != ref_word:
            mapping[hyp_word] = ref_word
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ground_truth")
    p.add_argument("predictions")
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--min-count", type=int, default=3)
    args = p.parse_args()

    gt, pred = read(args.ground_truth), read(args.predictions)
    keys = sorted(gt)
    missing = [k for k in keys if k not in pred]
    if missing:
        print(f"WARNING: {len(missing)} clips missing from predictions", file=sys.stderr)

    pairs = collections.Counter()
    for k in keys:
        for r, h in align_subs(gt[k], pred.get(k, "")):
            if r != h:
                pairs[(r, h)] += 1

    total_subs = sum(pairs.values())
    variant = {p_: n for p_, n in pairs.items() if similar(*p_, args.threshold)}
    variant_subs = sum(variant.values())

    base = corpus_wer(gt, pred, keys=keys)
    n_words = base.reference_words

    print("== substitution composition ==")
    print(f"  aligned substitutions:        {total_subs}")
    print(f"  near-identical spellings:     {variant_subs} "
          f"({variant_subs/max(total_subs,1)*100:.1f}% of substitutions)")
    print(f"  WER points they account for:  {variant_subs/n_words:.4f} "
          f"of {base.wer:.4f} total ({variant_subs/n_words/base.wer*100:.1f}%)")
    print(f"  similarity threshold: {args.threshold}, reference words: {n_words}")

    print("\n== are the references self-consistent? ==")
    bidir, one_way = [], []
    seen = set()
    for (r, h), n in sorted(variant.items(), key=lambda kv: -kv[1]):
        if (r, h) in seen:
            continue
        rev = variant.get((h, r), 0)
        seen.update({(r, h), (h, r)})
        (bidir if rev else one_way).append((r, h, n, rev))
    bd_mass = sum(n + rev for *_, n, rev in ((None, *b) for b in bidir))
    print(f"  pairs seen in BOTH directions: {len(bidir)} "
          f"(mass {sum(n+rev for _,_,n,rev in bidir)})")
    print(f"  pairs seen one way only:       {len(one_way)} "
          f"(mass {sum(n for _,_,n,_ in one_way)})")
    print("  bidirectional pairs cap what a global rule can recover:")
    for r, h, n, rev in bidir[:10]:
        print(f"    {r!r} -> {h!r}: {n} vs reverse {rev}  (net {n-rev:+d})")

    print("\n== two-fold cross-validated gain from a majority-vote map ==")
    half_a = keys[0::2]
    half_b = keys[1::2]
    gains = []
    for name, fit, score in (("A->B", half_a, half_b), ("B->A", half_b, half_a)):
        mapping = build_map(gt, pred, fit, args.min_count, args.threshold)
        before = corpus_wer(gt, pred, keys=score)
        after = corpus_wer(gt, pred, mapping=mapping, keys=score)
        gains.append(after.wer - before.wer)
        print(f"  fit {name}: map size {len(mapping):>3} | "
              f"{before.wer:.6f} -> {after.wer:.6f}  ({after.wer-before.wer:+.6f})")
    print(f"  mean held-out delta: {sum(gains)/len(gains):+.6f} "
          "(negative = the map helps)")

    full = build_map(gt, pred, keys, args.min_count, args.threshold)
    fitted = corpus_wer(gt, pred, mapping=full, keys=keys)
    print(f"\n  in-sample (OVERFIT, not claimable): {base.wer:.6f} -> "
          f"{fitted.wer:.6f} ({fitted.wer-base.wer:+.6f}), map size {len(full)}")

    print("\n== SAFE map: only spellings absent from the references ==")
    print("  A rewrite can only help when the hypothesis spelling is never")
    print("  correct. Everything else is context-dependent and must be learned.")
    safe_gains = []
    for name, fit, score in (("A->B", half_a, half_b), ("B->A", half_b, half_a)):
        mapping = build_map(gt, pred, fit, args.min_count, args.threshold, safe_only=True)
        before = corpus_wer(gt, pred, keys=score)
        after = corpus_wer(gt, pred, mapping=mapping, keys=score)
        safe_gains.append(after.wer - before.wer)
        print(f"  fit {name}: map size {len(mapping):>3} | "
              f"{before.wer:.6f} -> {after.wer:.6f}  ({after.wer-before.wer:+.6f})")
    print(f"  mean held-out delta: {sum(safe_gains)/len(safe_gains):+.6f} "
          "(negative = the map helps)")
    safe_full = build_map(gt, pred, keys, args.min_count, args.threshold, safe_only=True)
    print(f"  full-fold safe map ({len(safe_full)}): {safe_full}")


if __name__ == "__main__":
    main()
