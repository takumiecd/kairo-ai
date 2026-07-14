"""プロファイルのデータ構造 (docs/PROFILE.md §2)。

```
profile
├─ meta:      version, base_profile_id, total_units N（総確定文字数）
├─ explicit:  E（明示的シグナル・減衰なし）
│   entries: [ { input, surface, accept_count, reject_count, source } ]
├─ implicit:  （暗黙的シグナル・減衰あり）
│   ├─ unigram F:  surface -> { count, reading, last_used }
│   ├─ recency r:  直近 R 語のリングバッファ
│   ├─ domain d:   { code, prose, chat } 正規化ベクトル
│   └─ lang ℓ:     { ja_ratio, en_token_rate }
```

更新則そのものは :mod:`user_profile.builder` に置く。ここはデータ構造と
JSON シリアライズ/デシリアライズだけを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


DEFAULT_VERSION = 1
DOMAIN_LABELS: tuple[str, str, str] = ("code", "prose", "chat")


def _default_domain() -> dict[str, float]:
    return {label: 1.0 / len(DOMAIN_LABELS) for label in DOMAIN_LABELS}


def _default_lang() -> dict[str, float]:
    return {"ja_ratio": 0.0, "en_token_rate": 0.0}


@dataclass
class ExplicitEntry:
    """明示的シグナルの1エントリ。減衰しない。"""

    input: str
    surface: str
    accept_count: int = 0
    reject_count: int = 0
    source: str = "correction"  # "correction" | "registration"

    def to_dict(self) -> dict:
        return {
            "input": self.input,
            "surface": self.surface,
            "accept_count": self.accept_count,
            "reject_count": self.reject_count,
            "source": self.source,
        }

    @staticmethod
    def from_dict(data: dict) -> "ExplicitEntry":
        return ExplicitEntry(
            input=str(data["input"]),
            surface=str(data["surface"]),
            accept_count=int(data.get("accept_count", 0)),
            reject_count=int(data.get("reject_count", 0)),
            source=str(data.get("source", "correction")),
        )


@dataclass
class UnigramEntry:
    """暗黙的シグナル(unigram)の1エントリ。lazy decay で読む。"""

    count: float = 0.0
    reading: str | None = None
    last_used: int = 0  # この語が最後に触れられた時点の N

    def to_dict(self) -> dict:
        return {"count": self.count, "reading": self.reading, "last_used": self.last_used}

    @staticmethod
    def from_dict(data: dict) -> "UnigramEntry":
        return UnigramEntry(
            count=float(data.get("count", 0.0)),
            reading=data.get("reading"),
            last_used=int(data.get("last_used", 0)),
        )


@dataclass
class ProfileMeta:
    version: int = DEFAULT_VERSION
    base_profile_id: str | None = None
    total_units: int = 0  # N: 総確定文字数

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "base_profile_id": self.base_profile_id,
            "total_units": self.total_units,
        }

    @staticmethod
    def from_dict(data: dict) -> "ProfileMeta":
        return ProfileMeta(
            version=int(data.get("version", DEFAULT_VERSION)),
            base_profile_id=data.get("base_profile_id"),
            total_units=int(data.get("total_units", 0)),
        )


@dataclass
class Profile:
    """ユーザープロファイル u = (E, F, r, d, l, N)。"""

    meta: ProfileMeta = field(default_factory=ProfileMeta)
    explicit: list[ExplicitEntry] = field(default_factory=list)
    unigram: dict[str, UnigramEntry] = field(default_factory=dict)
    recency: list[str] = field(default_factory=list)
    domain: dict[str, float] = field(default_factory=_default_domain)
    lang: dict[str, float] = field(default_factory=_default_lang)

    def to_dict(self) -> dict:
        """JSON へシリアライズ可能な dict を作る(新しいコンテナのみで構成)。"""
        return {
            "meta": self.meta.to_dict(),
            "explicit": [entry.to_dict() for entry in self.explicit],
            "implicit": {
                "unigram": {
                    surface: entry.to_dict() for surface, entry in self.unigram.items()
                },
                "recency": list(self.recency),
                "domain": dict(self.domain),
                "lang": dict(self.lang),
            },
        }

    @staticmethod
    def from_dict(data: dict) -> "Profile":
        implicit = data.get("implicit", {})
        unigram_data = implicit.get("unigram", {})
        return Profile(
            meta=ProfileMeta.from_dict(data.get("meta", {})),
            explicit=[ExplicitEntry.from_dict(entry) for entry in data.get("explicit", [])],
            unigram={
                surface: UnigramEntry.from_dict(entry)
                for surface, entry in unigram_data.items()
            },
            recency=list(implicit.get("recency", [])),
            domain=dict(implicit.get("domain", _default_domain())),
            lang=dict(implicit.get("lang", _default_lang())),
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")

    @staticmethod
    def load_json(path: str | Path) -> "Profile":
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        return Profile.from_dict(data)
