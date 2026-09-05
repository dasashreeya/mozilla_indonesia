"""Error analysis for MDC-normalized ASR predictions.

Answers the questions that decide where to spend effort:

* What is the corpus WER, and how does it split into S/D/I?
* How much of the error is *orthographic* (casing, Javanese diacritics) rather
  than acoustic?  Those points are recoverable with text rules; acoustic errors
  are not.
* Which word pairs does the model actually confuse?
* Does a candidate post-processing rule help or hurt, measured rather than guessed?

Usage::

    PYTHONPATH=src python3 scripts/error_analysis.py ground_truth.csv predictions.csv
    PYTHONPATH=src python3 scripts/error_analysis.py gt.csv pred.csv --group-by language
    PYTHONPATH=src python3 scripts/error_analysis.py gt.csv pred.csv --top-confusions 50

Both CSVs need ``audio_filename`` and ``transcript`` columns.  Predictions are
aligned to the ground truth by ``audio_filename``.
"""

import argparse
import collections
import csv
import difflib
import re
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from mdc_asr.normalization import normalize_text
from mdc_asr.wer import ErrorCounts, corpus_errors


# --------------------------------------------------------------------------- io


def read_transcripts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("{} has no rows".format(path))
    for column in ("audio_filename", "transcript"):
        if column not in rows[0]:
            raise ValueError("{} is missing the {} column".format(path, column))
    return rows


def align(reference_path: Path, prediction_path: Path) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    references = read_transcripts(reference_path)
    predictions = {row["audio_filename"]: row.get("transcript") or "" for row in read_transcripts(prediction_path)}
    missing = [row["audio_filename"] for row in references if row["audio_filename"] not in predictions]
    if missing:
        raise ValueError("{} predictions missing (first: {})".format(len(missing), missing[0]))
    return references, predictions


# ------------------------------------------------------------------ text views


def strip_diacritics(text: str) -> str:
    """Drop combining marks: ``dhèwèké`` -> ``dheweke``."""

    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def casefold_view(text: str) -> str:
    return text.lower()


def plain_view(text: str) -> str:
    return strip_diacritics(text).lower()


# The diagnostics below relax BOTH sides.  They are not achievable scores: the
# real references are hidden and cannot be relaxed.  They bound how much WER is
# attributable to an orthographic convention mismatch, which tells you how much
# a text-side fix could ever be worth.
DIAGNOSTIC_VIEWS: Dict[str, Callable[[str], str]] = {
    "actual (scored as-is)": lambda text: text,
    "case-insensitive": casefold_view,
    "diacritic-insensitive": strip_diacritics,
    "case+diacritic-insensitive": plain_view,
}


# --------------------------------------------------- candidate post-processing


def collapse_word_repeats(text: str) -> str:
    return re.sub(r"\b(\w+)( \1\b)+", r"\1", text)


CANDIDATE_RULES: Dict[str, Callable[[str], str]] = {
    "as-is": lambda text: text,
    "lowercase-all": lambda text: text.lower(),
    "strip-diacritics": strip_diacritics,
    "hyphen-to-space": lambda text: text.replace("-", " "),
    "collapse-word-repeats": collapse_word_repeats,
}


# ---------------------------------------------------------------- computations


def score(references: List[Mapping[str, str]], predictions: Mapping[str, str],
          ref_view: Callable[[str], str], hyp_view: Callable[[str], str]) -> ErrorCounts:
    return corpus_errors(
        (ref_view(row.get("transcript") or ""), hyp_view(predictions[row["audio_filename"]]))
        for row in references
    )


def confusion_pairs(references: List[Mapping[str, str]],
                    predictions: Mapping[str, str]) -> Tuple[collections.Counter, int, int, int]:
    """Return substitution pairs plus counts of case-only and diacritic-only ones."""

    pairs: collections.Counter = collections.Counter()
    case_only = diacritic_only = total = 0
    for row in references:
        ref_words = normalize_text(row.get("transcript") or "").split()
        hyp_words = normalize_text(predictions[row["audio_filename"]]).split()
        matcher = difflib.SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            for ref_word, hyp_word in zip(ref_words[i1:i2], hyp_words[j1:j2]):
                pairs[(ref_word, hyp_word)] += 1
                total += 1
                if ref_word.lower() == hyp_word.lower():
                    case_only += 1
                elif plain_view(ref_word) == plain_view(hyp_word):
                    diacritic_only += 1
    return pairs, case_only, diacritic_only, total


def reference_casing_profile(references: List[Mapping[str, str]]) -> Tuple[int, int]:
    """Count mid-sentence capitalised reference words.

    The MDC normalizer lowercases only sentence-initial letters, so every one of
    these is a word the model must capitalise correctly or eat a substitution.
    """

    capitalised = words = 0
    for row in references:
        tokens = normalize_text(row.get("transcript") or "").split()
        words += len(tokens)
        capitalised += sum(1 for token in tokens[1:] if token[:1].isupper())
    return capitalised, words


# ---------------------------------------------------------------------- report


def _line(counts: ErrorCounts) -> str:
    return "WER={:.6f}  S={} D={} I={} N={}".format(
        counts.wer, counts.substitutions, counts.deletions, counts.insertions, counts.reference_words
    )


def report(reference_path: Path, prediction_path: Path, group_by: Optional[str],
           top_confusions: int) -> None:
    references, predictions = align(reference_path, prediction_path)
    identity = DIAGNOSTIC_VIEWS["actual (scored as-is)"]

    print("== corpus score ==")
    overall = score(references, predictions, identity, identity)
    print("  all: " + _line(overall))
    if group_by:
        groups: Dict[str, List[Mapping[str, str]]] = {}
        for row in references:
            groups.setdefault(row.get(group_by) or "<missing>", []).append(row)
        for name in sorted(groups):
            print("  {}: {}".format(name, _line(score(groups[name], predictions, identity, identity))))

    print("\n== where the error lives (relaxing BOTH sides; diagnostic only) ==")
    print("  {:<30} {:>9} {:>11}".format("view", "WER", "headroom"))
    for name, view in DIAGNOSTIC_VIEWS.items():
        counts = score(references, predictions, view, view)
        print("  {:<30} {:>9.6f} {:>+11.6f}".format(name, counts.wer, overall.wer - counts.wer))
    print("  Headroom is the WER attributable to an orthographic convention")
    print("  mismatch rather than to mishearing. It is an upper bound on what a")
    print("  text-side fix can recover, not a score you can claim.")

    capitalised, words = reference_casing_profile(references)
    share = capitalised / words if words else 0.0
    print("\n== reference casing profile ==")
    print("  mid-sentence capitalised reference words: {}/{} ({:.2%})".format(capitalised, words, share))
    print("  The scorer lowercases only sentence-initial letters, so blanket")
    print("  lowercasing of predictions costs roughly this share in substitutions.")

    print("\n== candidate post-processing rules (applied to predictions only) ==")
    print("  {:<24} {:>9} {:>11}".format("rule", "WER", "delta"))
    for name, rule in CANDIDATE_RULES.items():
        counts = score(references, predictions, identity, rule)
        print("  {:<24} {:>9.6f} {:>+11.6f}".format(name, counts.wer, counts.wer - overall.wer))
    print("  Negative delta = the rule helps. Validate on held-out folds before shipping.")

    pairs, case_only, diacritic_only, total = confusion_pairs(references, predictions)
    print("\n== substitutions ==")
    print("  counted: {}".format(total))
    if total:
        print("  differing only by case:      {} ({:.1%})".format(case_only, case_only / total))
        print("  differing only by diacritic: {} ({:.1%})".format(diacritic_only, diacritic_only / total))
    print("\n  top {} confusion pairs (reference -> hypothesis):".format(top_confusions))
    for (ref_word, hyp_word), count in pairs.most_common(top_confusions):
        if ref_word.lower() == hyp_word.lower():
            tag = "  [case]"
        elif plain_view(ref_word) == plain_view(hyp_word):
            tag = "  [diacritic]"
        else:
            tag = ""
        print("    {:>4}  {!r} -> {!r}{}".format(count, ref_word, hyp_word, tag))

    worst = sorted(
        references,
        key=lambda row: -corpus_errors([(row.get("transcript") or "", predictions[row["audio_filename"]])]).errors,
    )[:10]
    print("\n== 10 worst clips ==")
    for row in worst:
        counts = corpus_errors([(row.get("transcript") or "", predictions[row["audio_filename"]])])
        print("    {}  errors={} N={}".format(row["audio_filename"], counts.errors, counts.reference_words))
        print("      ref: {}".format(normalize_text(row.get("transcript") or "")))
        print("      hyp: {}".format(normalize_text(predictions[row["audio_filename"]])))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, help="Ground truth CSV")
    parser.add_argument("prediction", type=Path, help="Predictions CSV")
    parser.add_argument("--group-by", help="Reference column for per-group diagnostic WER (e.g. language)")
    parser.add_argument("--top-confusions", type=int, default=30)
    args = parser.parse_args(argv)
    report(args.reference, args.prediction, args.group_by, args.top_confusions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
