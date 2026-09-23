"""Compile observation JSON into one canonical Soramimic Score document."""

from __future__ import annotations

import argparse
from pathlib import Path

from .document import compile_score, dump, from_linked_observations
from .ir import IntermediateRepresentation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Stage 3 intermediate JSON")
    parser.add_argument("--output", type=Path, required=True, help="Soramimic Score JSON")
    parser.add_argument(
        "--linked-input",
        action="store_true",
        help="Input already contains Stage 3 links; compile it without decoding",
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


if __name__ == "__main__":
    raise SystemExit(main())
