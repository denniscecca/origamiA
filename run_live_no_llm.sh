#!/usr/bin/env bash
set -euo pipefail
python3 -m http.server 7777 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
python3 upu_auto_loop_live_v2.py --cycles 999999 --sleep 0.7 --render --frames 18 --no-llm --no-history-scenes
