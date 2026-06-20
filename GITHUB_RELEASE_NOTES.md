# vLLM Web - GitHub Upload Notes

## What to Upload

Upload this repository as-is, excluding files ignored by `.gitignore`.

Important included files:

- `app/main.py` - FastAPI backend and vLLM process manager
- `app/static/` - browser UI
- `requirements.txt` - Python dependencies
- `data/profiles.example.json` - safe starter profile
- `scripts/run_ui.sh` - local launcher
- `README.md` - user setup and usage guide

Ignored local-only files:

- `data/profiles.json` - may contain API keys or private paths
- `data/logs/` - runtime logs
- `review_artifacts/` - local screenshots/test artifacts
- `.venv/` and Python cache files

## Suggested GitHub Description

Local-first web app for configuring, validating, launching, and exporting `vllm serve` profiles.

## Suggested First Release Text

Initial public test package for vLLM Web:

- Wizard mode for guided setup
- Expert mode for full vLLM options
- Live command preview
- Downloadable startup script
- vLLM server start/stop/restart controls
- GPU telemetry when `nvidia-smi` is available
- Ray/systemd/Spark-Ray export helpers

Security note: run on `127.0.0.1` or behind a protected private network. Do not expose this app directly to the public Internet.
