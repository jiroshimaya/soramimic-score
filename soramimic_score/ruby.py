"""Conservative display ruby using the selected line pronunciation."""

from __future__ import annotations

import re

from .japanese import katakana


_NEEDS_RUBY = re.compile(r"[一-龯々〆ヵヶA-Za-zＡ-Ｚａ-ｚ0-9０-９]")
_TRAILING_KANA = re.compile(r"[ぁ-ゖァ-ヿー]+$")


def _token_segments(surface: str, reading: str) -> list[dict[str, str]]:
    if not _NEEDS_RUBY.search(surface):
        return [{"text": surface, "reading": ""}]
    suffix = _TRAILING_KANA.search(surface)
    if suffix and suffix.start() and katakana(reading).endswith(katakana(suffix.group())):
        base = surface[:suffix.start()]
        base_reading = reading[:len(reading) - len(suffix.group())]
        if base_reading:
            return [{"text": base, "reading": base_reading},
                    {"text": suffix.group(), "reading": ""}]
    return [{"text": surface, "reading": reading}]


def ruby_segments(text: str, selected_kana: str) -> list[dict[str, str]]:
    """Annotate tokens only when their readings match the selected pronunciation.

    If a nonstandard singing pronunciation changes token alignment, omit ruby
    rather than attach a possibly wrong reading to a token or the whole line.
    """
    if not _NEEDS_RUBY.search(text):
        return [{"text": text, "reading": ""}]
    try:
        from soramimic_yomi import get_tokens
        tokens = get_tokens(text, apply_rules=True)
    except Exception:
        return [{"text": text, "reading": ""}]
    readings = [token["pronunciation"] for token in tokens]
    if "".join(token["surface_form"] for token in tokens) != text:
        return [{"text": text, "reading": ""}]
    if katakana("".join(readings)) != katakana(selected_kana):
        # Keep ruby when the selected singing pronunciation changes one token.
        # The surrounding dictionary readings anchor that token unambiguously.
        for index in range(len(tokens)):
            prefix = "".join(readings[:index])
            suffix = "".join(readings[index + 1:])
            if (katakana(selected_kana).startswith(katakana(prefix))
                    and katakana(selected_kana).endswith(katakana(suffix))):
                replacement = selected_kana[len(prefix):len(selected_kana) - len(suffix) if suffix else None]
                if replacement:
                    readings[index] = replacement
                    break
        else:
            return [{"text": text, "reading": ""}]
    if katakana("".join(readings)) == katakana(selected_kana):
        return [part for token, reading in zip(tokens, readings, strict=True)
                for part in _token_segments(token["surface_form"], reading)]
    return [{"text": text, "reading": ""}]
