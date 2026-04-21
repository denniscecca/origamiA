# UPU Origami Interactive 4D Field

Interactive 4D→3D UPU/Origami field visualizer with optional local LLM control loop.

## Files

- `upu_origami_interactive_4d_config.py` — renders the 4D→3D interactive Plotly HTML scene from JSON config.
- `upu_auto_loop.py` — synchronous optimization loop: render, compute metrics, call LLM or fallback, save history.
- `upu_auto_loop_live_v2.py` — continuous non-blocking live loop: field keeps running while LLM calls happen in background.
- `upu_llm_controller_v2.py` — one-shot controller that asks a llama.cpp/Msty-compatible endpoint for the next config.
- `upu_field_config.json` — base numeric config.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Render one interactive HTML scene

```bash
python3 upu_origami_interactive_4d_config.py \
  --config upu_field_config.json \
  --out upu_scene.html \
  --frames 80
```

Then open `upu_scene.html` in a browser.

## 2) Run live loop without LLM

This is the safest first test because it does not require llama.cpp or Msty.

Terminal 1:

```bash
python3 -m http.server 7777
```

Terminal 2:

```bash
python3 upu_auto_loop_live_v2.py \
  --cycles 999999 \
  --sleep 0.7 \
  --render \
  --frames 18 \
  --no-llm \
  --no-history-scenes
```

Open:

```text
http://127.0.0.1:7777/upu_live_viewer.html
```

## 3) Run live loop with local LLM

Start a llama.cpp/OpenAI-compatible server first, for example on:

```text
http://127.0.0.1:8080/v1/chat/completions
```

Then run:

```bash
python3 upu_auto_loop_live_v2.py \
  --api http://127.0.0.1:8080/v1/chat/completions \
  --model local \
  --cycles 999999 \
  --sleep 0.7 \
  --render \
  --frames 18 \
  --llm-every 8 \
  --llm-timeout 45 \
  --no-history-scenes
```

## 4) One-shot LLM config proposal

```bash
python3 upu_llm_controller_v2.py \
  --api http://127.0.0.1:8080/v1/chat/completions \
  --model local \
  --config upu_field_config.json \
  --out-config upu_field_config_next.json \
  --goal "Return only JSON. Increase uStar return, lower temperature slightly, keep mixed states alive." \
  --manual-fallback \
  --render
```

## Config keys

```json
{
  "phi_gain": 1.0,
  "k_gain": 1.0,
  "temperature": 0.50,
  "ustar_pull": 0.18,
  "four_u_radius": 1.80,
  "golden_lock": 1.61803398875,
  "z_coherence_gain": 1.0,
  "mixed_bridge_gain": 0.72,
  "rotation_speed": 1.0,
  "phase_twist": 0.55
}
```

## Recommended GitHub start

```bash
git init
git add .
git commit -m "Initial UPU Origami field release"
```

Choose a license before publishing, for example MIT for permissive open source.

## References / Zenodo

This project is connected to the broader UPU / ECR / Unity research line documented here:

- **ECR / Unity field framework**  
  https://doi.org/10.5281/zenodo.19652991

- **Extended formulation and numerical experiments**  
  https://doi.org/10.5281/zenodo.19643454
