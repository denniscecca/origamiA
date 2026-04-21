#!/usr/bin/env bash
set -euo pipefail
python3 upu_origami_interactive_4d_config.py --config upu_field_config.json --out upu_scene.html --frames 80
printf '\nOpen: upu_scene.html\n'
