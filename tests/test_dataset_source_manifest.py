import json
import tempfile
import unittest
from pathlib import Path

from dataset.source_manifest import load_manifest
from dataset.source_manifest import slugify


class DatasetSourceManifestTest(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Natsume Soseki: Wagahai wa Neko"), "Natsume_Soseki_Wagahai_wa_Neko")

    def test_load_manifest_accepts_object_with_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sources.json"
            path.write_text(
                json.dumps({"sources": [{"source": "file.txt"}]}),
                encoding="utf-8",
            )

            sources = load_manifest(path)

        self.assertEqual(sources, [{"source": "file.txt"}])

    def test_load_manifest_accepts_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sources.json"
            path.write_text(json.dumps([{"source": "file.txt"}]), encoding="utf-8")

            sources = load_manifest(path)

        self.assertEqual(sources, [{"source": "file.txt"}])


if __name__ == "__main__":
    unittest.main()
