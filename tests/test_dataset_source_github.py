import unittest
from unittest import mock

from dataset import source_github


class ParseRepoArgTest(unittest.TestCase):
    def test_parses_owner_slash_repo(self):
        self.assertEqual(source_github.parse_repo_arg("octocat/Hello-World"), ("octocat", "Hello-World"))

    def test_parses_https_url(self):
        self.assertEqual(
            source_github.parse_repo_arg("https://github.com/octocat/Hello-World"),
            ("octocat", "Hello-World"),
        )

    def test_strips_git_suffix(self):
        self.assertEqual(
            source_github.parse_repo_arg("https://github.com/octocat/Hello-World.git"),
            ("octocat", "Hello-World"),
        )

    def test_rejects_missing_slash(self):
        with self.assertRaises(ValueError):
            source_github.parse_repo_arg("not-a-repo")


class CheckLicenseAllowedTest(unittest.TestCase):
    def test_allows_mit(self):
        source_github.check_license_allowed("MIT")

    def test_allows_apache(self):
        source_github.check_license_allowed("Apache-2.0")

    def test_rejects_gpl(self):
        with self.assertRaises(source_github.LicenseNotAllowedError):
            source_github.check_license_allowed("GPL-3.0")

    def test_rejects_missing_license(self):
        with self.assertRaises(source_github.LicenseNotAllowedError):
            source_github.check_license_allowed(None)

    def test_rejects_noassertion(self):
        with self.assertRaises(source_github.LicenseNotAllowedError):
            source_github.check_license_allowed("NOASSERTION")


class StripPiiTest(unittest.TestCase):
    def test_removes_email(self):
        self.assertNotIn("@", source_github.strip_pii("connect at dev@example.com please"))

    def test_removes_url(self):
        self.assertNotIn("http", source_github.strip_pii("see https://example.com/issue/123 for detail"))

    def test_removes_long_token(self):
        cleaned = source_github.strip_pii("token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 leaked")
        self.assertNotIn("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345", cleaned)

    def test_preserves_japanese_text(self):
        text = "バグを修正した。READMEを更新した。"
        self.assertEqual(source_github.strip_pii(text), text)

    def test_removes_mention(self):
        cleaned = source_github.strip_pii("@takumiecd さん確認お願いします")
        self.assertNotIn("@takumiecd", cleaned)

    def test_removes_identity_bearing_commit_trailer(self):
        cleaned = source_github.strip_pii(
            "バグを修正した。\nCo-authored-by: Example Person <person@example.com>"
        )
        self.assertNotIn("Example Person", cleaned)
        self.assertIn("バグを修正した。", cleaned)


class BuildCorpusTextTest(unittest.TestCase):
    def test_combines_readme_and_commits_without_pii(self):
        corpus = source_github.build_corpus_text(
            readme="このプロジェクトはIMEです。連絡先 owner@example.com",
            commit_messages=["バグを修正した", "READMEを更新した https://example.com/pr/1"],
        )

        self.assertIn("このプロジェクトはIMEです。", corpus)
        self.assertIn("バグを修正した", corpus)
        self.assertNotIn("@example.com", corpus)
        self.assertNotIn("https://", corpus)


class IngestRepoTest(unittest.TestCase):
    def test_rejects_disallowed_license_before_fetching_commits(self):
        responses = {
            "/repos/acme/tool": {
                "license": {"spdx_id": "GPL-3.0"},
                "default_branch": "main",
            },
        }

        def fake_request(path, token):
            return responses[path]

        with mock.patch.object(source_github, "github_request", side_effect=fake_request):
            with self.assertRaises(source_github.LicenseNotAllowedError):
                source_github.ingest_repo(
                    "acme/tool",
                    token=None,
                    max_commits=10,
                    min_chars=1,
                    max_chars=80,
                    max_units=None,
                )

    def test_ingests_allowed_repo(self):
        responses = {
            "/repos/acme/tool": {
                "license": {"spdx_id": "MIT"},
                "default_branch": "main",
            },
            "/repos/acme/tool/commits/main": {"sha": "deadbeef"},
            "/repos/acme/tool/license": {
                "encoding": "base64",
                "content": source_github.base64.b64encode(
                    "MIT License\nCopyright Example".encode("utf-8")
                ).decode("ascii"),
            },
            "/repos/acme/tool/readme": {
                "encoding": "base64",
                "content": source_github.base64.b64encode(
                    "IMEツールのREADMEです。".encode("utf-8")
                ).decode("ascii"),
            },
            "/repos/acme/tool/commits?sha=main&per_page=10&page=1": [
                {"commit": {"message": "バグを修正した。連絡先はowner@example.comまで。"}},
                {"commit": {"message": "READMEを更新した。"}},
            ],
        }

        def fake_request(path, token):
            return responses[path]

        with mock.patch.object(source_github, "github_request", side_effect=fake_request):
            text_units, manifest = source_github.ingest_repo(
                "acme/tool",
                token=None,
                max_commits=10,
                min_chars=1,
                max_chars=80,
                max_units=None,
            )

        self.assertEqual(manifest["repo"], "acme/tool")
        self.assertEqual(manifest["license_spdx_id"], "MIT")
        self.assertIn("MIT License", manifest["license_text"])
        self.assertEqual(manifest["legal_basis"], "repository_license")
        self.assertEqual(manifest["usage_scope"], "profile_training_and_inference")
        self.assertEqual(manifest["commit_sha"], "deadbeef")
        self.assertEqual(manifest["commit_messages"], 2)
        joined = "".join(text_units)
        self.assertNotIn("@example.com", joined)
        self.assertIn("バグを修正した", joined)


if __name__ == "__main__":
    unittest.main()
