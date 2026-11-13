# Development rules

A Palmimo app. Read README.md first.

- Use Python >=3.12 and uv. Never use pip or bare python.
- Run from this directory: `uv sync`, `uv run pytest -q`, `uv run ruff check .`, `uv run mypy`.
- Add dependencies with `uv add`; keep uv.lock committed and current.
- Type all function signatures and use Google-style English docstrings.
- Keep comments and diagnostics in English. Lines the robot speaks are Japanese.
- Reach the robot through the `palmimo_sdk` classes. Do not open the servo bus independently.
- Rehearse with `--dry-run` before running on hardware.
- The result is a rough reference value. Never describe it as a validated acuity measurement.
- Never commit API keys, shared passwords, device credentials or personal results.
- Upstream SDK rules: https://github.com/Jizai-inc/palmimo-devkit/blob/main/AGENTS.md
