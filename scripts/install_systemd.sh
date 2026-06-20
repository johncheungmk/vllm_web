#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE=/etc/systemd/system/vllm-web.service
sudo tee "$SERVICE" > /dev/null <<EOF
[Unit]
Description=vLLM Web
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8899
Restart=on-failure
RestartSec=5
Environment=VLLM_UI_DATA=$APP_DIR/data

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-web
systemctl status vllm-web --no-pager
