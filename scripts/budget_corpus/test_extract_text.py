"""Tests for budget HTML → plain text extraction."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from budget_corpus.extract_text import extracted_main_inner_html, html_to_plain_text, normalize_whitespace


class NormalizeWhitespaceTests(unittest.TestCase):
    def test_collapses_newlines_and_tabs(self) -> None:
        raw = "if you found a good\n          job, worked hard"
        self.assertEqual(
            normalize_whitespace(raw),
            "if you found a good job, worked hard",
        )

    def test_nbsp(self) -> None:
        self.assertEqual(normalize_whitespace("a\u00a0b"), "a b")


class ExtractTextTests(unittest.TestCase):
    def test_pretty_printed_paragraph_has_no_internal_newlines(self) -> None:
        html = """<!DOCTYPE html><html><body><main id="wb-main">
        <div id="wb-cont"><p>For generations, one of the foundational promises was that if you found a good
          job, worked hard, and saved money, you could afford a home.</p></div>
        </main></body></html>"""
        plain = html_to_plain_text(html)
        blocks = [b for b in re.split(r"\n{2,}", plain.strip()) if b]
        self.assertTrue(blocks)
        for block in blocks:
            self.assertNotIn("\n", block, msg=f"block contains newline: {block[:80]!r}")

    def test_merged_html_fragment_has_no_double_spaces(self) -> None:
        html = """<!DOCTYPE html><html><body><main>
        <p>Les travailleurs et les  entreprises du Canada ont fait preuve d'une
          résilience remarquable.</p>
        </main></body></html>"""
        inner = extracted_main_inner_html(html)
        self.assertNotIn("  ", inner)
        self.assertIn("les entreprises", inner)

    def test_chap1_en_regression(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "data/corpora/federal-budget/raw/2024/en/chap1-en.html"
        if not sample.exists():
            self.skipTest(f"missing fixture {sample}")
        plain = html_to_plain_text(sample.read_text(encoding="utf-8"))
        blocks = [b for b in re.split(r"\n{2,}", plain.strip()) if b]
        self.assertGreater(len(blocks), 10)
        bad = [b for b in blocks if "\n" in b]
        self.assertEqual(bad, [], msg=f"{len(bad)} blocks still contain internal newlines")


if __name__ == "__main__":
    unittest.main()
