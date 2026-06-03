import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from dataset.source_tatoeba import parse_tatoeba_sentences
from dataset.source_tatoeba import read_sentences_text


class DatasetSourceTatoebaTest(unittest.TestCase):
    def test_parse_tatoeba_sentences_filters_japanese(self):
        text = "\n".join(
            [
                "1\teng\tHello.",
                "2\tjpn\tこんにちは。",
                "3\tjpn\t今日はテストです。",
            ]
        )

        sentences = parse_tatoeba_sentences(text, lang="jpn")

        self.assertEqual(sentences, ["こんにちは。", "今日はテストです。"])

    def test_parse_tatoeba_sentences_deduplicates(self):
        text = "1\tjpn\tこんにちは。\n2\tjpn\tこんにちは。"

        sentences = parse_tatoeba_sentences(text, lang="jpn")

        self.assertEqual(sentences, ["こんにちは。"])

    def test_parse_tatoeba_sentences_supports_comma_fallback(self):
        text = "1,jpn,こんにちは。"

        sentences = parse_tatoeba_sentences(text, lang="jpn")

        self.assertEqual(sentences, ["こんにちは。"])

    def test_read_sentences_text_reads_tar_bz2(self):
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:bz2") as archive:
            data = "1\tjpn\tこんにちは。".encode("utf-8")
            info = tarfile.TarInfo("sentences.csv")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sentences.tar.bz2"
            path.write_bytes(payload.getvalue())

            text = read_sentences_text(str(path))

        self.assertEqual(text, "1\tjpn\tこんにちは。")


if __name__ == "__main__":
    unittest.main()
