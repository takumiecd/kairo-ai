"""更新作用素 U (docs/PROFILE.md §2, §5)。

実行時 (``apply_commit``/``apply_correction``/``apply_rejection`` を
feedback イベントごとに呼ぶ) と学習時 (仮想ユーザーストリームを
``apply_commit`` へ時系列に流す) の両方から、**同一のこのクラス**を使う。
これが「推論時と同じ状況を学習時に再現する」ための唯一の接着剤
(PROFILE.md 冒頭・§6)。
"""

from __future__ import annotations

import re

from user_profile.schema import DOMAIN_LABELS
from user_profile.schema import ExplicitEntry
from user_profile.schema import Profile
from user_profile.schema import UnigramEntry


DEFAULT_UNIGRAM_CAP = 10_000
DEFAULT_RECENCY_SIZE = 500
DEFAULT_HALF_LIFE = 100_000  # H: 半減期(総確定文字数単位)

_ASCII_WORD_CHAR_RE = re.compile(r"[A-Za-z0-9_]")
_CODE_SYMBOLS = set("{}[]();=<>+-*/%&|^~`$#\\")
_CHAT_MARKS = ("!", "?", "！", "？", "w", "笑", "ｗ")


def _char_class(char: str) -> str:
    codepoint = ord(char)
    if 0x3040 <= codepoint <= 0x309F:
        return "hiragana"
    if 0x30A0 <= codepoint <= 0x30FF:
        return "katakana"
    if 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF:
        return "kanji"
    if _ASCII_WORD_CHAR_RE.match(char):
        return "ascii_word"
    return "other"


def split_words(text: str) -> list[str]:
    """簡易な単語分割器。

    形態素解析は行わない。連続する漢字/カタカナ/ひらがなの境界と、ASCII
    単語(``[A-Za-z0-9_]+``)の境界だけで区切る。句読点・空白などの
    ``other`` クラスの文字は単語として拾わず、区切り記号としてのみ働く。
    """
    words: list[str] = []
    current: list[str] = []
    current_class: str | None = None
    for char in text:
        char_class = _char_class(char)
        if char_class == "other":
            if current:
                words.append("".join(current))
                current = []
                current_class = None
            continue
        if current_class is not None and char_class != current_class:
            words.append("".join(current))
            current = []
        current_class = char_class
        current.append(char)
    if current:
        words.append("".join(current))
    return words


def infer_domain(text: str) -> str:
    """domain ヒントが与えられなかったときの簡易推定。

    厳密な分類器ではない。ASCII 比率が高くコード記号を含めば ``code``、
    短くて感嘆符・「w」等の口語マークを含めば ``chat``、それ以外は
    ``prose`` とする。学習時は persona から domain を直接渡す方が正確。
    """
    if not text:
        return "prose"
    ascii_ratio = sum(1 for char in text if ord(char) < 128) / len(text)
    code_symbols = sum(1 for char in text if char in _CODE_SYMBOLS)
    if ascii_ratio >= 0.5 and code_symbols > 0:
        return "code"
    if len(text) <= 12 and any(mark in text for mark in _CHAT_MARKS):
        return "chat"
    return "prose"


class ProfileBuilder:
    """プロファイル更新作用素。イベントごとに ``profile`` を書き換える。"""

    def __init__(
        self,
        profile: Profile | None = None,
        unigram_cap: int = DEFAULT_UNIGRAM_CAP,
        recency_size: int = DEFAULT_RECENCY_SIZE,
        half_life: int = DEFAULT_HALF_LIFE,
    ) -> None:
        self.profile = profile if profile is not None else Profile()
        self.unigram_cap = unigram_cap
        self.recency_size = recency_size
        self.half_life = half_life

    # ------------------------------------------------------------------
    # イベント
    # ------------------------------------------------------------------

    def apply_commit(
        self,
        surface_text: str,
        reading: str | None = None,
        domain: str | None = None,
    ) -> None:
        """確定イベント。unigram/recency/domain/lang/N を更新する。"""
        if not surface_text:
            return

        words = split_words(surface_text)
        units = len(surface_text)
        total_before = self.profile.meta.total_units
        total_after = total_before + units
        self.profile.meta.total_units = total_after

        # surface_text がちょうど1語のときだけ reading をその語に紐づける。
        # (feedback イベントは1確定=1語相当のことが多く、複数語にまたがる
        # 場合はどの語の読みか自明でないため付与しない。)
        word_reading = reading if len(words) == 1 else None
        for word in words:
            self._touch_unigram(word, total_after, word_reading)
        self._enforce_unigram_cap()

        self._push_recency(words)
        self._update_domain(surface_text, domain, units, total_before, total_after)
        self._update_lang(surface_text, words, units, total_before, total_after)

    def apply_correction(self, input_text: str, surface: str) -> None:
        """修正イベント(候補選び直し・登録) -> explicit へ、accept_count++。"""
        entry = self._find_explicit(input_text, surface)
        if entry is None:
            entry = ExplicitEntry(input=input_text, surface=surface, source="correction")
            self.profile.explicit.append(entry)
        entry.accept_count += 1

    def apply_rejection(self, input_text: str, surface: str) -> None:
        """却下イベント -> explicit の reject_count++。"""
        entry = self._find_explicit(input_text, surface)
        if entry is None:
            entry = ExplicitEntry(input=input_text, surface=surface, source="correction")
            self.profile.explicit.append(entry)
        entry.reject_count += 1

    # ------------------------------------------------------------------
    # 読み出し
    # ------------------------------------------------------------------

    def decayed_count(self, surface: str) -> float:
        """lazy decay 込みの unigram カウント c̃(w; N) を計算する。"""
        entry = self.profile.unigram.get(surface)
        if entry is None:
            return 0.0
        return self._decay(entry.count, entry.last_used, self.profile.meta.total_units)

    def snapshot(self) -> dict:
        """軽量な(deep copy でない)シリアライズ形式のスナップショットを返す。"""
        return self.profile.to_dict()

    # ------------------------------------------------------------------
    # 内部実装
    # ------------------------------------------------------------------

    def _decay(self, count: float, last_used: int, now: int) -> float:
        elapsed = max(0, now - last_used)
        return count * (2.0 ** (-elapsed / self.half_life))

    def _find_explicit(self, input_text: str, surface: str) -> ExplicitEntry | None:
        for entry in self.profile.explicit:
            if entry.input == input_text and entry.surface == surface:
                return entry
        return None

    def _touch_unigram(self, word: str, now: int, reading: str | None) -> None:
        unigram = self.profile.unigram
        entry = unigram.get(word)
        if entry is None:
            entry = UnigramEntry(count=0.0, reading=reading, last_used=now)
            unigram[word] = entry
        else:
            # 書き込み時に一旦 last_used 時点まで減衰させてから加算する。
            # decayed_count() は読み出し時にも同じ式を適用するので、
            # ここで先に畳み込んでおいても結果は等価(冪等)。
            entry.count = self._decay(entry.count, entry.last_used, now)
            if reading is not None:
                entry.reading = reading
        entry.count += 1.0
        entry.last_used = now

    def _enforce_unigram_cap(self) -> None:
        unigram = self.profile.unigram
        overflow = len(unigram) - self.unigram_cap
        if overflow <= 0:
            return
        now = self.profile.meta.total_units
        ordered = sorted(
            unigram.items(),
            key=lambda item: self._decay(item[1].count, item[1].last_used, now),
        )
        for surface, _ in ordered[:overflow]:
            del unigram[surface]

    def _push_recency(self, words: list[str]) -> None:
        recency = self.profile.recency
        recency.extend(words)
        if len(recency) > self.recency_size:
            del recency[: len(recency) - self.recency_size]

    def _update_domain(
        self,
        surface_text: str,
        domain: str | None,
        units: int,
        total_before: int,
        total_after: int,
    ) -> None:
        if domain is None:
            domain = infer_domain(surface_text)
        if domain not in DOMAIN_LABELS or total_after <= 0:
            return
        vector = self.profile.domain
        for label in DOMAIN_LABELS:
            vector[label] = vector.get(label, 0.0) * total_before / total_after
        vector[domain] = vector.get(domain, 0.0) + units / total_after

    def _update_lang(
        self,
        surface_text: str,
        words: list[str],
        units: int,
        total_before: int,
        total_after: int,
    ) -> None:
        if total_after <= 0:
            return
        lang = self.profile.lang

        ja_chars = sum(1 for char in surface_text if _char_class(char) in ("hiragana", "katakana", "kanji"))
        lang["ja_ratio"] = (lang.get("ja_ratio", 0.0) * total_before + ja_chars) / total_after

        ascii_words = sum(1 for word in words if _char_class(word[0]) == "ascii_word")
        en_token_rate_event = ascii_words / len(words) if words else 0.0
        lang["en_token_rate"] = (
            lang.get("en_token_rate", 0.0) * total_before + en_token_rate_event * units
        ) / total_after
