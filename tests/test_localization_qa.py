import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from localization_qa import analyze_rows, run  # noqa: E402


class AnalyzeRowsTests(unittest.TestCase):
    def issue_types(self, rows, **kwargs):
        return {issue.issue_type for issue in analyze_rows(rows, **kwargs)}

    def test_detects_chinese_residual(self):
        rows = [{"id": "1", "source_zh": "订单已确认。", "translation_en": "您的 order is confirmed."}]
        self.assertIn("chinese_residual", self.issue_types(rows))

    def test_detects_glossary_mismatch(self):
        rows = [{"id": "1", "source_zh": "我们会处理退款。", "translation_en": "We will give your money back."}]
        issues = analyze_rows(rows, glossary={"退款": "refund"})
        self.assertTrue(any(issue.issue_type == "glossary_mismatch" for issue in issues))

    def test_detects_duplicate_translation(self):
        rows = [
            {"id": "1", "source_zh": "你好。", "translation_en": "Hello!"},
            {"id": "2", "source_zh": "欢迎。", "translation_en": " hello "},
        ]
        issues = analyze_rows(rows)
        duplicates = [issue for issue in issues if issue.issue_type == "duplicate_translation"]
        self.assertEqual(1, len(duplicates))
        self.assertEqual("2", duplicates[0].record_id)

    def test_clean_record_has_no_issue_with_wide_ratio_range(self):
        rows = [{"id": "1", "source_zh": "您好，请问有什么可以帮您？", "translation_en": "Hello, how can I help you?"}]
        issues = analyze_rows(rows, min_length_ratio=0.1, max_length_ratio=2.0)
        self.assertEqual([], issues)


class ReportIntegrationTests(unittest.TestCase):
    def test_run_writes_csv_and_json_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.csv"
            report_path = root / "report.csv"
            summary_path = root / "summary.json"
            glossary_path = root / "glossary.json"

            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "source_zh", "translation_en"])
                writer.writeheader()
                writer.writerow({"id": "1", "source_zh": "请申请退款。", "translation_en": ""})
            glossary_path.write_text(json.dumps({"退款": "refund"}, ensure_ascii=False), encoding="utf-8")

            args = Namespace(
                input=input_path,
                output=report_path,
                summary=summary_path,
                glossary=glossary_path,
                source_column="source_zh",
                translation_column="translation_en",
                id_column="id",
                min_length_ratio=0.25,
                max_length_ratio=1.5,
                fail_on="none",
            )
            summary, issues = run(args)

            self.assertEqual(1, summary["rows_analyzed"])
            self.assertEqual(1, summary["issue_count"])
            self.assertEqual("missing_translation", issues[0].issue_type)
            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(1, json.loads(summary_path.read_text(encoding="utf-8"))["issue_count"])


if __name__ == "__main__":
    unittest.main()
