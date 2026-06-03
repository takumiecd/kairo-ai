"""Typing noise and romaji variant augmentation."""

from __future__ import annotations

from dataclasses import dataclass
import random


ROMAJI_VARIANTS = [
    ("shi", "si"),
    ("chi", "ti"),
    ("tsu", "tu"),
    ("fu", "hu"),
    ("jya", "ja"),
    ("ja", "jya"),
    ("shu", "syu"),
    ("cho", "tyo"),
    ("wo", "o"),
    ("nn", "n"),
]

QWERTY_NEIGHBORS = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "ersfcx",
    "e": "wsdr",
    "f": "rtgdvc",
    "g": "tyhfbv",
    "h": "yujgnb",
    "i": "ujko",
    "j": "uikhnm",
    "k": "ioljm",
    "l": "opk",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "wedxza",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}


@dataclass(frozen=True)
class NoisyInput:
    text: str
    noise: str


@dataclass(frozen=True)
class NoiseConfig:
    typo_rate: float = 0.03
    deletion_rate: float = 0.02
    duplication_rate: float = 0.02
    swap_rate: float = 0.02
    romaji_variant_rate: float = 0.35


class InputNoiser:
    def __init__(self, config: NoiseConfig | None = None, seed: int = 0) -> None:
        self.config = config or NoiseConfig()
        self.random = random.Random(seed)

    def augment(self, text: str, n: int) -> list[NoisyInput]:
        variants = [NoisyInput(text=text, noise="none")]
        operations = [
            ("romaji_variant", self.apply_romaji_variant),
            ("keyboard_typo", self.apply_keyboard_typo),
            ("deletion", self.apply_deletion),
            ("duplication", self.apply_duplication),
            ("swap", self.apply_swap),
        ]

        attempts = 0
        seen = {text}
        while len(variants) < n + 1 and attempts < max(20, n * 10):
            noise_name, operation = self.random.choice(operations)
            noisy = operation(text)
            attempts += 1
            if noisy != text and noisy not in seen:
                seen.add(noisy)
                variants.append(NoisyInput(text=noisy, noise=noise_name))

        return variants

    def augment_segments(self, segments: list[tuple[str, bool]], n: int) -> list[NoisyInput]:
        """Augment only mutable segments, preserving literal/code spans."""
        text = "".join(segment for segment, _mutable in segments)
        variants = [NoisyInput(text=text, noise="none")]
        mutable_indexes = [index for index, (_segment, mutable) in enumerate(segments) if mutable]
        if not mutable_indexes:
            return variants

        operations = [
            ("romaji_variant", self.apply_romaji_variant),
            ("keyboard_typo", self.apply_keyboard_typo),
            ("deletion", self.apply_deletion),
            ("duplication", self.apply_duplication),
            ("swap", self.apply_swap),
        ]

        attempts = 0
        seen = {text}
        while len(variants) < n + 1 and attempts < max(20, n * 10):
            noise_name, operation = self.random.choice(operations)
            segment_index = self.random.choice(mutable_indexes)
            noisy_segments = [segment for segment, _mutable in segments]
            noisy_segments[segment_index] = operation(noisy_segments[segment_index])
            noisy = "".join(noisy_segments)
            attempts += 1
            if noisy != text and noisy not in seen:
                seen.add(noisy)
                variants.append(NoisyInput(text=noisy, noise=noise_name))

        return variants

    def apply_romaji_variant(self, text: str) -> str:
        output = text
        for src, dst in ROMAJI_VARIANTS:
            if src in output and self.random.random() < self.config.romaji_variant_rate:
                output = output.replace(src, dst, 1)
        return output

    def apply_keyboard_typo(self, text: str) -> str:
        chars = list(text)
        indexes = [i for i, char in enumerate(chars) if char.lower() in QWERTY_NEIGHBORS]
        if not indexes:
            return text

        changed = False
        for index in indexes:
            if self.random.random() < self.config.typo_rate:
                char = chars[index]
                replacement = self.random.choice(QWERTY_NEIGHBORS[char.lower()])
                chars[index] = replacement.upper() if char.isupper() else replacement
                changed = True

        if not changed:
            index = self.random.choice(indexes)
            char = chars[index]
            replacement = self.random.choice(QWERTY_NEIGHBORS[char.lower()])
            chars[index] = replacement.upper() if char.isupper() else replacement

        return "".join(chars)

    def apply_deletion(self, text: str) -> str:
        indexes = [i for i, char in enumerate(text) if char.isalpha()]
        if not indexes:
            return text
        index = self._choose_index(indexes, self.config.deletion_rate)
        return text[:index] + text[index + 1 :]

    def apply_duplication(self, text: str) -> str:
        indexes = [i for i, char in enumerate(text) if char.isalpha()]
        if not indexes:
            return text
        index = self._choose_index(indexes, self.config.duplication_rate)
        return text[:index] + text[index] + text[index:]

    def apply_swap(self, text: str) -> str:
        indexes = [
            i
            for i in range(len(text) - 1)
            if text[i].isalpha() and text[i + 1].isalpha()
        ]
        if not indexes:
            return text
        index = self._choose_index(indexes, self.config.swap_rate)
        chars = list(text)
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
        return "".join(chars)

    def _choose_index(self, indexes: list[int], rate: float) -> int:
        candidates = [index for index in indexes if self.random.random() < rate]
        return self.random.choice(candidates or indexes)
