import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from dataset.source_text import clean_aozora_text
from dataset.source_text import extract_text_units
from dataset.source_text import read_source_text


class DatasetSourceTextTest(unittest.TestCase):
    def test_clean_aozora_text_removes_ruby_and_notes(self):
        raw = "\n".join(
            [
                "タイトル",
                "-------------------------------------------------------",
                "凡例",
                "-------------------------------------------------------",
                "吾輩《わがはい》は猫である。［＃改ページ］",
                "｜名前はまだ無い。",
                "底本：テスト",
            ]
        )

        cleaned = clean_aozora_text(raw)

        self.assertEqual(cleaned, "吾輩は猫である。\n名前はまだ無い。")

    def test_extract_text_units_splits_sentences(self):
        units = extract_text_units(
            "これは最初の文です。これは次の文です！短い。",
            min_chars=5,
            max_chars=80,
        )

        self.assertEqual(units, ["これは最初の文です。", "これは次の文です！"])

    def test_read_source_text_reads_zip_txt(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, mode="w") as archive:
            archive.writestr("sample.txt", "吾輩は猫である。".encode("utf-8"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.zip"
            path.write_bytes(payload.getvalue())

            text = read_source_text(str(path))

        self.assertEqual(text, "吾輩は猫である。")


if __name__ == "__main__":
    unittest.main()
