import bz2
import tempfile
import unittest
from pathlib import Path

from dataset.source_wikipedia import clean_wikipedia_text
from dataset.source_wikipedia import extract_wikipedia_units
from dataset.source_wikipedia import iter_wikipedia_pages


SAMPLE_XML = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>テスト</title>
    <ns>0</ns>
    <revision>
      <text>'''テスト'''は[[日本語]]の記事です。{{Infobox}}
== 見出し ==
これは現代的な文章です。[[Category:テスト]]</text>
    </revision>
  </page>
  <page>
    <title>リダイレクト</title>
    <ns>0</ns>
    <redirect title="テスト" />
    <revision><text>#REDIRECT [[テスト]]</text></revision>
  </page>
  <page>
    <title>ノート:テスト</title>
    <ns>1</ns>
    <revision><text>本文ではない。</text></revision>
  </page>
</mediawiki>"""


class DatasetSourceWikipediaTest(unittest.TestCase):
    def test_clean_wikipedia_text_removes_markup(self):
        cleaned = clean_wikipedia_text(
            "'''テスト'''は[[日本語|日本語]]です。{{Infobox}}\n"
            "[[Category:テスト]]\n"
            "[https://example.com 表示名]"
        )

        self.assertIn("テストは日本語です。", cleaned)
        self.assertIn("表示名", cleaned)
        self.assertNotIn("Category", cleaned)
        self.assertNotIn("Infobox", cleaned)

    def test_iter_wikipedia_pages_skips_redirect_and_non_article(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.xml"
            path.write_text(SAMPLE_XML, encoding="utf-8")

            pages = list(iter_wikipedia_pages(str(path)))

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0][0], "テスト")

    def test_extract_wikipedia_units_reads_bz2_dump(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.xml.bz2"
            path.write_bytes(bz2.compress(SAMPLE_XML.encode("utf-8")))

            units = extract_wikipedia_units(
                str(path),
                max_units=4,
                min_chars=8,
                max_chars=80,
            )

        self.assertIn("テストは日本語の記事です。", units)
        self.assertIn("これは現代的な文章です。", units)


if __name__ == "__main__":
    unittest.main()
