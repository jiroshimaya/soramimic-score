# Repository agent rules

- The delivery branch is `main`; implementation is complete after its pull request is merged.
- Keep the primary `soramimic-score` checkout on `main` and perform changes in a task-specific linked worktree.
- Do not commit audio, MIDI, lyrics, user uploads, generated media, model weights, credentials, or evaluation truth.
- Run `python scripts/check_repository_boundary.py` and the full test suite before every commit.
