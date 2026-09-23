"""Compile observation JSON into one canonical Soramimic Score document."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .document import compile_score, dump, from_linked_observations
from .ir import IntermediateRepresentation


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        return analyze_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="音源の解析: soramimic-score analyze --help")
    parser.add_argument("--input", type=Path, required=True, help="Observation JSON")
    parser.add_argument("--output", type=Path, required=True, help="Soramimic Score JSON")
    parser.add_argument(
        "--linked-input",
        action="store_true",
        help="Input already contains correspondence links; compile without decoding",
    )
    args = parser.parse_args()
    document = IntermediateRepresentation.from_json(args.input.read_text(encoding="utf-8"))
    result = (
        from_linked_observations(document)
        if args.linked_input
        else compile_score(document)
    )
    dump(result, args.output)
    print(
        f"{len(result.score.synthesis_plan)} slots; "
        f"{len(result.score.unresolved_unit_ids)} unresolved units"
    )
    return 0


def analyze_main(argv) -> int:
    from .audio import analyze_audio
    from .models import ModelConfig
    parser = argparse.ArgumentParser(description="歌唱音源から歌詞つき音符JSONを生成")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sheetsage-model", type=Path, required=True)
    parser.add_argument("--sheetsage-base", type=Path, required=True)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--ctc-model", default="reazon-research/japanese-wav2vec2-base-rs35kh")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--lyrics", type=Path, help="UTF-8歌詞ファイル（1行1フレーズ）")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.audio.resolve() == args.output.resolve() or (
        args.lyrics is not None and args.lyrics.resolve() == args.output.resolve()
    ):
        parser.error("出力先は入力ファイルと別にしてください")
    try:
        lyrics = (tuple(line for line in args.lyrics.read_text(encoding="utf-8").splitlines()
                        if line.strip()) if args.lyrics else None)
        config = ModelConfig(args.sheetsage_model, args.sheetsage_base, args.whisper_model,
                             args.ctc_model, args.device, args.local_files_only)
        score = analyze_audio(args.audio, model_config=config, lyrics=lyrics)
        dump(score, args.output)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"解析に失敗しました: {exc}\n")
    print(f"{len(score.score.synthesis_plan)} slots; saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
