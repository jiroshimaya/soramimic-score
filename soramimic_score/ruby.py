"""Conservative display ruby using the selected line pronunciation."""

from __future__ import annotations

import re

from .japanese import katakana


_NEEDS_RUBY = re.compile(r"[一-龯々〆ヵヶA-Za-zＡ-Ｚａ-ｚ0-9０-９]")


def ruby_segments(text: str, selected_kana: str) -> list[dict[str, str]]:
    """Annotate tokens only when their readings match the selected pronunciation.

    If a nonstandard singing pronunciation changes token alignment, put the
    selected reading over the full line rather than display a wrong token ruby.
    """
    if not _NEEDS_RUBY.search(text):
        return [{"text": text, "reading": ""}]
    from soramimic_yomi import get_tokens

    try:
        tokens = get_tokens(text, apply_rules=True)
    except Exception:
        return [{"text": text, "reading": selected_kana}]
    readings = [token["pronunciation"] for token in tokens]
    if ("".join(token["surface_form"] for token in tokens) == text
            and katakana("".join(readings)) == katakana(selected_kana)):
        return [{"text": token["surface_form"],
                 "reading": reading if _NEEDS_RUBY.search(token["surface_form"]) else ""}
                for token, reading in zip(tokens, readings, strict=True)]
    return [{"text": text, "reading": selected_kana}]
