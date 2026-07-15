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


class FetchIssueTextsTest(unittest.TestCase):
    def test_skips_pull_requests_but_includes_any_author(self):
        page1 = [
            {
                "title": "変換精度が落ちる",
                "body": "特定の入力でCERが悪化する",
                "user": {"login": "acme"},
            },
            {"title": "PRです", "body": "説明", "pull_request": {"url": "x"}, "user": {"login": "acme"}},
            {
                "title": "他人が書いたissue",
                "body": "第三者の報告",
                "user": {"login": "someone-else"},
            },
        ]

        def fake_request(path, token):
            self.assertIn("/repos/acme/tool/issues?state=all", path)
            if "page=1" in path:
                return page1
            return []

        with mock.patch.object(source_github, "github_request", side_effect=fake_request):
            texts = source_github.fetch_issue_texts("acme", "tool", None, max_issues=50)

        self.assertEqual(len(texts), 2)
        self.assertIn("変換精度が落ちる", texts[0])
        self.assertIn("他人が書いたissue", texts[1])

    def test_stops_at_max_issues(self):
        page1 = [
            {"title": f"issue{i}", "body": "本文", "user": {"login": "acme"}}
            for i in range(3)
        ]

        def fake_request(path, token):
            return page1

        with mock.patch.object(source_github, "github_request", side_effect=fake_request):
            texts = source_github.fetch_issue_texts("acme", "tool", None, max_issues=2)

        self.assertEqual(len(texts), 2)


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
        self.assertEqual(manifest["commit_sha"], "deadbeef")
        self.assertEqual(manifest["commit_messages"], 2)
        joined = "".join(text_units)
        self.assertNotIn("@example.com", joined)
        self.assertIn("バグを修正した", joined)


if __name__ == "__main__":
    unittest.main()
