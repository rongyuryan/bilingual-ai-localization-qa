"""Rule-based quality checks for Chinese-to-English AI localization output."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")
REPEATED_WORD_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z'-]*)\s+\1\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_WORD_PATTERN = re.compile(r"[^a-z0-9']+")
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class Issue:
    row_number: int
    record_id: str
    issue_type: str
    severity: str
    message: str
    source_text: str
    translation_text: str


def load_glossary(path: Path | None) -> dict[str, str]:
    """Load a JSON object that maps Chinese terms to required English terms."""
    if path is None:
        return {}

    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise ValueError("Glossary must be a JSON object of string-to-string entries.")
    return {key.strip(): value.strip() for key, value in data.items() if key.strip() and value.strip()}


def normalized_translation(text: str) -> str:
    """Normalize an English translation before duplicate comparison."""
    lowered = text.casefold().strip()
    return NON_WORD_PATTERN.sub(" ", lowered).strip()


def source_unit_count(text: str) -> int:
    """Count CJK characters plus non-CJK word groups in a source sentence."""
    cjk_count = len(CJK_PATTERN.findall(text))
    without_cjk = CJK_PATTERN.sub(" ", text)
    other_words = len(re.findall(r"[A-Za-z0-9]+", without_cjk))
    return cjk_count + other_words


def translation_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text))


def punctuation_problems(text: str) -> list[str]:
    """Return descriptions of unmatched brackets and quotation marks."""
    problems: list[str] = []
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    for opening, closing in pairs:
        if text.count(opening) != text.count(closing):
            problems.append(f"unbalanced {opening}{closing}")
    if text.count('"') % 2:
        problems.append('unbalanced double quotation marks')
    if text.count("“") != text.count("”"):
        problems.append("unbalanced curly quotation marks")
    return problems


def terminal_punctuation_problem(source: str, translation: str) -> str | None:
    source = source.rstrip()
    translation = translation.rstrip()
    if source.endswith(("?", "？")) and not translation.endswith("?"):
        return "source is a question but the translation does not end with a question mark"
    if source.endswith(("!", "！")) and not translation.endswith("!"):
        return "source is an exclamation but the translation does not end with an exclamation mark"
    return None


def check_record(
    *,
    row_number: int,
    record_id: str,
    source: str,
    translation: str,
    glossary: Mapping[str, str],
    min_length_ratio: float,
    max_length_ratio: float,
) -> list[Issue]:
    """Run all checks that depend on one source/translation pair."""
    issues: list[Issue] = []

    def add(issue_type: str, severity: str, message: str) -> None:
        issues.append(
            Issue(
                row_number=row_number,
                record_id=record_id,
                issue_type=issue_type,
                severity=severity,
                message=message,
                source_text=source,
                translation_text=translation,
            )
        )

    if not source:
        add("missing_source", "high", "Source text is empty.")
    if not translation:
        add("missing_translation", "high", "English translation is empty.")
        return issues

    residual = sorted(set(CJK_PATTERN.findall(translation)))
    if residual:
        preview = "".join(residual[:8])
        add(
            "chinese_residual",
            "high",
            f"Translation still contains Chinese characters: {preview}",
        )

    repeated = sorted({match.group(0) for match in REPEATED_WORD_PATTERN.finditer(translation)})
    if repeated:
        add(
            "repeated_word",
            "medium",
            f"Possible accidental repeated word sequence: {', '.join(repeated)}",
        )

    for problem in punctuation_problems(translation):
        add("unbalanced_punctuation", "medium", problem.capitalize() + ".")

    terminal_problem = terminal_punctuation_problem(source, translation)
    if terminal_problem:
        add("terminal_punctuation", "low", terminal_problem.capitalize() + ".")

    translation_lower = translation.casefold()
    for source_term, required_term in glossary.items():
        if source_term in source and required_term.casefold() not in translation_lower:
            add(
                "glossary_mismatch",
                "medium",
                f'Source contains "{source_term}" but translation does not contain required term "{required_term}".',
            )

    source_units = source_unit_count(source)
    if source_units:
        ratio = translation_word_count(translation) / source_units
        if ratio < min_length_ratio or ratio > max_length_ratio:
            add(
                "length_outlier",
                "low",
                f"English-word/source-unit ratio is {ratio:.2f}; expected {min_length_ratio:.2f}-{max_length_ratio:.2f}.",
            )

    return issues


def analyze_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    source_column: str = "source_zh",
    translation_column: str = "translation_en",
    id_column: str = "id",
    glossary: Mapping[str, str] | None = None,
    min_length_ratio: float = 0.25,
    max_length_ratio: float = 1.50,
) -> list[Issue]:
    """Analyze CSV-like records and return issues in stable row order."""
    glossary = glossary or {}
    issues: list[Issue] = []
    duplicate_groups: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)

    for row_number, row in enumerate(rows, start=2):
        source = (row.get(source_column) or "").strip()
        translation = WHITESPACE_PATTERN.sub(" ", (row.get(translation_column) or "").strip())
        record_id = (row.get(id_column) or str(row_number - 1)).strip()
        issues.extend(
            check_record(
                row_number=row_number,
                record_id=record_id,
                source=source,
                translation=translation,
                glossary=glossary,
                min_length_ratio=min_length_ratio,
                max_length_ratio=max_length_ratio,
            )
        )
        normalized = normalized_translation(translation)
        if normalized:
            duplicate_groups[normalized].append((row_number, record_id, source, translation))

    for group in duplicate_groups.values():
        if len(group) < 2:
            continue
        first_id = group[0][1]
        for row_number, record_id, source, translation in group[1:]:
            issues.append(
                Issue(
                    row_number=row_number,
                    record_id=record_id,
                    issue_type="duplicate_translation",
                    severity="medium",
                    message=f'Translation duplicates record "{first_id}" after normalization.',
                    source_text=source,
                    translation_text=translation,
                )
            )

    return sorted(issues, key=lambda item: (item.row_number, -SEVERITY_ORDER[item.severity], item.issue_type))


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")
        return list(reader), list(reader.fieldnames)


def write_issue_report(path: Path, issues: Iterable[Issue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Issue.__dataclass_fields__)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            writer.writerow(asdict(issue))


def build_summary(rows_analyzed: int, issues: Sequence[Issue]) -> dict[str, object]:
    return {
        "rows_analyzed": rows_analyzed,
        "issue_count": len(issues),
        "issues_by_severity": dict(sorted(Counter(item.severity for item in issues).items())),
        "issues_by_type": dict(sorted(Counter(item.issue_type for item in issues).items())),
        "records_with_issues": len({item.record_id for item in issues}),
    }


def write_summary(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def should_fail(issues: Sequence[Issue], fail_on: str) -> bool:
    if fail_on == "none":
        return False
    threshold = SEVERITY_ORDER[fail_on]
    return any(SEVERITY_ORDER[issue.severity] >= threshold for issue in issues)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Chinese-to-English AI localization output and generate QA reports."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file.")
    parser.add_argument("--output", type=Path, default=Path("qa_report.csv"), help="Issue report CSV.")
    parser.add_argument("--summary", type=Path, help="Summary JSON; defaults beside the CSV report.")
    parser.add_argument("--glossary", type=Path, help="Optional Chinese-to-English glossary JSON.")
    parser.add_argument("--source-column", default="source_zh")
    parser.add_argument("--translation-column", default="translation_en")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--min-length-ratio", type=float, default=0.25)
    parser.add_argument("--max-length-ratio", type=float, default=1.50)
    parser.add_argument(
        "--fail-on",
        choices=("none", "medium", "high"),
        default="none",
        help="Return exit code 1 when issues at or above this severity exist.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[dict[str, object], list[Issue]]:
    if args.min_length_ratio < 0 or args.max_length_ratio <= args.min_length_ratio:
        raise ValueError("Length-ratio thresholds must satisfy 0 <= minimum < maximum.")

    rows, fieldnames = read_csv(args.input)
    required_columns = {args.source_column, args.translation_column}
    missing = sorted(required_columns.difference(fieldnames))
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {', '.join(missing)}")

    issues = analyze_rows(
        rows,
        source_column=args.source_column,
        translation_column=args.translation_column,
        id_column=args.id_column,
        glossary=load_glossary(args.glossary),
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
    )
    summary = build_summary(len(rows), issues)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    write_issue_report(args.output, issues)
    write_summary(summary_path, summary)
    return summary, issues


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        summary, issues = run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Analyzed {summary['rows_analyzed']} rows; found {summary['issue_count']} issues "
        f"across {summary['records_with_issues']} records."
    )
    print(f"CSV report: {args.output}")
    print(f"JSON summary: {args.summary or args.output.with_suffix('.summary.json')}")
    return 1 if should_fail(issues, args.fail_on) else 0


if __name__ == "__main__":
    raise SystemExit(main())
