# Bilingual AI Localization QA Checker

![tests](https://github.com/rongyuryan/bilingual-ai-localization-qa/actions/workflows/tests.yml/badge.svg)

A dependency-free Python command-line tool that reviews Chinese-to-English AI localization output and produces an auditable issue report. It turns recurring manual quality checks into a repeatable workflow for localization, conversational AI, and content operations teams.

## What it checks

- Missing source text or English translations
- Chinese characters left in English output
- Required terminology from a configurable glossary
- Duplicate translations after case and punctuation normalization
- Accidental repeated English words
- Unbalanced brackets and quotation marks
- Question and exclamation mark mismatches
- Unusually short or long translations based on a configurable length ratio

Each issue includes the CSV row, record ID, severity, explanation, source text, and translation. The tool also writes a JSON summary grouped by severity and issue type.

## Quick start

Requires Python 3.10 or later and uses only the Python standard library.

```bash
python localization_qa.py \
  --input examples/dialogues.csv \
  --glossary examples/glossary.json \
  --output reports/issues.csv
```

Windows PowerShell:

```powershell
python .\localization_qa.py `
  --input .\examples\dialogues.csv `
  --glossary .\examples\glossary.json `
  --output .\reports\issues.csv
```

Expected console output:

```text
Analyzed 9 rows; found 7 issues across 7 records.
CSV report: reports/issues.csv
JSON summary: reports/issues.summary.json
```

## Input format

The default CSV headers are:

```csv
id,source_zh,translation_en
EP01-001,您好，请问有什么可以帮您？,"Hello, how can I help you?"
```

Custom headers are supported:

```bash
python localization_qa.py --input data.csv \
  --source-column chinese \
  --translation-column english \
  --id-column line_id
```

The optional glossary is a UTF-8 JSON object:

```json
{
  "退款": "refund",
  "订单": "order",
  "客户服务": "customer support"
}
```

## Useful options

```text
--min-length-ratio 0.25   Minimum English-word/source-unit ratio
--max-length-ratio 1.50   Maximum English-word/source-unit ratio
--summary PATH            Custom JSON summary path
--fail-on medium          Exit with code 1 for medium- or high-severity issues
```

`--fail-on` makes the checker usable in a batch pipeline or CI workflow. Input/configuration errors return exit code 2.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers residual Chinese detection, terminology checks, duplicate detection, a clean record, and end-to-end report generation.

## Project structure

```text
localization_qa.py          Core checks, report writers, and CLI
examples/dialogues.csv      Sample Chinese-to-English data
examples/glossary.json      Sample required terminology
tests/test_localization_qa.py
reports/                    Generated reports (ignored by Git)
```

## Possible next steps

- Add a small web interface for non-technical reviewers
- Highlight exact character spans that caused an issue
- Support per-client terminology and severity rules
- Compare rule-based findings with human reviewer decisions
- Add precision and recall evaluation on a labelled dataset
