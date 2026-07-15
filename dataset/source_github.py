"""GitHubリポジトリのライセンスを確認し、OKなら日本語テキスト(README+コミット
メッセージ+任意でissue)をプレーンテキストコーパスへ変換する。

主な用途は `dataset.profile_stream` の `--persona name=path:domain:plain` で使う
engineerペルソナ用コーパス。ライセンスが許可リストに無いリポジトリは拒否し、
コミットの著者名・メールアドレス等の個人情報は取り込まない(本文中に紛れ込んだ
メール・URL・@メンション・トークン様文字列も機械的に除去する)。ソースコード
ファイル本体やコード内コメントは対象外(日本語のかな漢字変換学習という目的に
対して価値が薄いため)。issueはPRを除くtitle/bodyを取得する
(`--include-issues`、既定オフ)。issueは投稿者を問わず取得する。リポジトリの
LICENSEがissueまで及ぶかは不明瞭だが、機械学習(情報解析目的)での利用は
著作権法30条の4により権利者の許諾なく行える、というのがここでの法的根拠
(`docs/DATA_POLICY.md` 参照)。

使い方::

    python -m dataset.source_github \\
        --repo octocat/Hello-World \\
        --output data/external/github/octocat_hello_world.txt \\
        --manifest data/external/github/octocat_hello_world.manifest.json
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen

from dataset.source_text import extract_text_units
from dataset.source_text import normalize_lines


API_ROOT = "https://api.github.com"

# 複製・改変・別ライセンスでの再配布が明示的に許可されている(かつ要求が著作権
# 表示保持程度に留まる)ライセンスのみ。コピーレフト系(GPL/LGPL/AGPL)や
# NC/ND系、LICENSE未設定(NOASSERTION)は対象外。
ALLOWED_LICENSES = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "CC0-1.0",
    "Unlicense",
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
URL_RE = re.compile(r"https?://\S+")
TOKEN_LIKE_RE = re.compile(r"\b[A-Za-z0-9_-]{24,}\b")
MENTION_RE = re.compile(r"@[A-Za-z0-9_-]+")


class LicenseNotAllowedError(RuntimeError):
    """指定リポジトリのライセンスが許可リストに無い場合に送出する。"""


@dataclass(frozen=True)
class RepoLicenseInfo:
    spdx_id: str | None
    default_branch: str


def github_request(path: str, token: str | None):
    url = path if path.startswith("http") else f"{API_ROOT}{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "kairo-ai-dataset-tool",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_repo_arg(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        raw = raw.split("github.com/", 1)[-1]
    raw = raw.rstrip("/")
    if raw.endswith(".git"):
        raw = raw[: -len(".git")]
    if "/" not in raw:
        raise ValueError(f"invalid --repo value (expected owner/repo): {raw!r}")
    owner, repo = raw.split("/", 1)
    return owner, repo


def fetch_repo_info(owner: str, repo: str, token: str | None) -> RepoLicenseInfo:
    payload = github_request(f"/repos/{owner}/{repo}", token)
    license_field = payload.get("license") or {}
    return RepoLicenseInfo(
        spdx_id=license_field.get("spdx_id"),
        default_branch=payload["default_branch"],
    )


def check_license_allowed(spdx_id: str | None) -> None:
    if not spdx_id or spdx_id in {"NOASSERTION", "NONE"} or spdx_id not in ALLOWED_LICENSES:
        raise LicenseNotAllowedError(
            f"license {spdx_id!r} is not on the allow-list {sorted(ALLOWED_LICENSES)}; "
            "confirm the repository's LICENSE file manually before using it."
        )


def fetch_latest_commit_sha(owner: str, repo: str, branch: str, token: str | None) -> str:
    payload = github_request(f"/repos/{owner}/{repo}/commits/{branch}", token)
    return payload["sha"]


def fetch_readme_text(owner: str, repo: str, token: str | None) -> str:
    try:
        payload = github_request(f"/repos/{owner}/{repo}/readme", token)
    except HTTPError as error:
        if error.code == 404:
            return ""
        raise
    if payload.get("encoding") != "base64":
        return ""
    return base64.b64decode(payload["content"]).decode("utf-8", errors="replace")


def fetch_commit_messages(
    owner: str, repo: str, branch: str, token: str | None, max_commits: int
) -> list[str]:
    messages: list[str] = []
    page = 1
    per_page = min(100, max_commits) if max_commits > 0 else 0
    while per_page > 0 and len(messages) < max_commits:
        payload = github_request(
            f"/repos/{owner}/{repo}/commits?sha={branch}&per_page={per_page}&page={page}",
            token,
        )
        if not payload:
            break
        for entry in payload:
            message = entry.get("commit", {}).get("message", "")
            if message:
                messages.append(message)
            if len(messages) >= max_commits:
                break
        if len(payload) < per_page:
            break
        page += 1
    return messages


def fetch_issue_texts(
    owner: str, repo: str, token: str | None, max_issues: int
) -> list[str]:
    """Issueのtitle+bodyを取得する(PRは``/issues``に混ざって返るため除外)。

    投稿者はリポジトリオーナーに限定しない(``--include-issues``で明示的に
    オプトインした場合のみ、第三者投稿分も含めて取得する。法的根拠は
    著作権法30条の4(情報解析目的の利用)。詳細は``docs/DATA_POLICY.md``)。
    """
    texts: list[str] = []
    page = 1
    per_page = min(100, max_issues) if max_issues > 0 else 0
    while per_page > 0 and len(texts) < max_issues:
        payload = github_request(
            f"/repos/{owner}/{repo}/issues?state=all&per_page={per_page}&page={page}",
            token,
        )
        if not payload:
            break
        for entry in payload:
            if "pull_request" in entry:
                continue
            title = entry.get("title") or ""
            body = entry.get("body") or ""
            combined = "\n".join(part for part in (title, body) if part)
            if combined:
                texts.append(combined)
            if len(texts) >= max_issues:
                break
        if len(payload) < per_page:
            break
        page += 1
    return texts


def strip_pii(text: str) -> str:
    text = EMAIL_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = TOKEN_LIKE_RE.sub("", text)
    return text


def build_corpus_text(
    readme: str,
    commit_messages: list[str],
    issue_texts: list[str] | None = None,
) -> str:
    blocks = [strip_pii(readme)]
    blocks.extend(strip_pii(message) for message in commit_messages)
    blocks.extend(strip_pii(text) for text in issue_texts or [])
    return normalize_lines("\n".join(blocks))


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def write_text_corpus(path: Path, text_units: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for unit in text_units:
            file.write(unit + "\n")


def ingest_repo(
    repo_arg: str,
    token: str | None,
    max_commits: int,
    min_chars: int,
    max_chars: int,
    max_units: int | None,
    include_issues: bool = False,
    max_issues: int = 0,
) -> tuple[list[str], dict]:
    owner, repo = parse_repo_arg(repo_arg)
    info = fetch_repo_info(owner, repo, token)
    check_license_allowed(info.spdx_id)
    commit_sha = fetch_latest_commit_sha(owner, repo, info.default_branch, token)
    readme = fetch_readme_text(owner, repo, token)
    commit_messages = fetch_commit_messages(
        owner, repo, info.default_branch, token, max_commits
    )
    issue_texts = (
        fetch_issue_texts(owner, repo, token, max_issues) if include_issues else []
    )
    corpus_text = build_corpus_text(readme, commit_messages, issue_texts)
    text_units = extract_text_units(
        corpus_text, max_units=max_units, min_chars=min_chars, max_chars=max_chars
    )
    manifest = {
        "repo": f"{owner}/{repo}",
        "license_spdx_id": info.spdx_id,
        "default_branch": info.default_branch,
        "commit_sha": commit_sha,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "commit_messages": len(commit_messages),
        "issues": len(issue_texts),
        "has_readme": bool(readme),
        "text_units": len(text_units),
    }
    return text_units, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo または GitHub URL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--token", default=None, help="未指定なら環境変数 GITHUB_TOKEN を使う"
    )
    parser.add_argument("--max-commits", type=int, default=300)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=80)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument(
        "--include-issues",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="issue(PR以外、投稿者は問わない)のtitle/bodyも取り込む",
    )
    parser.add_argument("--max-issues", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("GITHUB_TOKEN")
    text_units, manifest = ingest_repo(
        args.repo,
        token=token,
        max_commits=args.max_commits,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_units=args.max_units,
        include_issues=args.include_issues,
        max_issues=args.max_issues,
    )
    write_text_corpus(args.output, text_units)
    manifest["output"] = str(args.output)
    if args.manifest:
        write_manifest(args.manifest, manifest)
    print(
        f"repo={manifest['repo']} license={manifest['license_spdx_id']} "
        f"text_units={len(text_units)}"
    )


if __name__ == "__main__":
    main()
