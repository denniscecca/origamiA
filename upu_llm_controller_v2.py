#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UPU LLM Controller v2 — robusto per llama.cpp / Msty.

Fix rispetto alla versione prima:
- se il modello non restituisce JSON, stampa la risposta raw
- accetta anche JSON dentro markdown
- supporta /completion fallback se vuoi
- ha --manual-fallback: crea config_next anche se il modello parla troppo
- prompt più aggressivo: "ONLY JSON"

Uso:
python3 upu_llm_controller_v2.py \
  --api http://127.0.0.1:8080/v1/chat/completions \
  --model local \
  --goal "modifica poco: aumenta ustar_pull di 0.05, abbassa temperature di 0.05, mantieni mixed_bridge_gain stabile" \
  --render
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ALLOWED = {
    "phi_gain": (0.5, 2.0),
    "k_gain": (0.5, 2.0),
    "temperature": (0.0, 1.0),
    "ustar_pull": (0.0, 0.8),
    "four_u_radius": (0.8, 2.6),
    "golden_lock": (1.2, 2.2),
    "z_coherence_gain": (0.5, 2.0),
    "mixed_bridge_gain": (0.25, 1.4),
    "rotation_speed": (0.2, 2.5),
    "phase_twist": (0.1, 1.5),
}

DEFAULT_DELTA = {
    "ustar_pull": 0.05,
    "temperature": -0.05,
    "mixed_bridge_gain": 0.0,
}

def clamp(k, v):
    lo, hi = ALLOWED[k]
    return max(lo, min(hi, float(v)))

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def extract_json(text):
    text = text.strip()

    # Rimuove markdown code fences
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Prova parsing diretto
    try:
        return json.loads(text)
    except Exception:
        pass

    # Cerca primo oggetto JSON bilanciato in modo semplice
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Cerca pattern ```json {...} ```
    m = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))

    raise ValueError("Nessun oggetto JSON valido trovato nella risposta del modello.")

def make_manual_fallback(current, goal):
    nxt = dict(current)
    low = goal.lower()
    if "ustar" in low or "ricomposizione" in low:
        nxt["ustar_pull"] = clamp("ustar_pull", float(nxt.get("ustar_pull", 0.18)) + 0.05)
    if "abbassa temperature" in low or "bassa temperature" in low or "temperatura" in low:
        nxt["temperature"] = clamp("temperature", float(nxt.get("temperature", 0.5)) - 0.05)
    if "mixed" in low and "stabile" in low:
        nxt["mixed_bridge_gain"] = clamp("mixed_bridge_gain", float(nxt.get("mixed_bridge_gain", 0.72)))
    return nxt

def call_llm(api, model, current_config, goal, max_tokens):
    schema = {k: current_config.get(k, None) for k in ALLOWED}

    system = (
        "You are a numeric JSON controller. "
        "Return ONLY valid JSON. No prose. No markdown. No explanations. "
        "Your entire response must start with { and end with }. "
        "Use exactly these keys: "
        + ", ".join(ALLOWED.keys()) + ". "
        "Keep changes small, normally within +/-0.10. "
        "Keep golden_lock near 1.61803398875 unless explicitly needed."
    )

    user = (
        "Current config:\n"
        + json.dumps(schema, indent=2)
        + "\n\nAllowed ranges:\n"
        + json.dumps(ALLOWED, indent=2)
        + "\n\nGoal:\n"
        + goal
        + "\n\nReturn ONLY this JSON object with numeric values, no text."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 0.8,
        "max_tokens": max_tokens,
        "stream": False,
    }

    req = urllib.request.Request(
        api,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"]

def render_html(render_script, config_path, html_path):
    cmd = [sys.executable, render_script, "--config", config_path, "--out", html_path]
    print("\n=== Rendering HTML ===")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print(f"HTML: {html_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="local")
    ap.add_argument("--config", default="./upu_field_config.json")
    ap.add_argument("--out-config", default="./upu_field_config_next.json")
    ap.add_argument("--goal", default="stabilizza il delta, aumenta il ritorno a uStar e mantieni vivi gli stati mixed")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-script", default="./upu_origami_interactive_4d_config.py")
    ap.add_argument("--html", default="./upu_scene_llm.html")
    ap.add_argument("--max-tokens", type=int, default=350)
    ap.add_argument("--manual-fallback", action="store_true", help="Se il modello non dà JSON, applica una micro-modifica manuale coerente col goal.")
    args = ap.parse_args()

    current = load_json(args.config)

    try:
        raw = call_llm(args.api, args.model, current, args.goal, args.max_tokens)
        print("=== LLM raw response ===")
        print(raw)
        proposed = extract_json(raw)

        next_cfg = dict(current)
        for k in ALLOWED:
            if k in proposed:
                next_cfg[k] = clamp(k, proposed[k])

    except Exception as e:
        print("\nERRORE: il modello ha risposto, ma non con JSON valido.")
        print(f"Dettaglio: {e}")

        if not args.manual_fallback:
            print("\nRilancia con --manual-fallback oppure usa il prompt più secco sotto:")
            print('Goal consigliato: "Return only JSON. ustar_pull=+0.05, temperature=-0.05, keep mixed_bridge_gain unchanged."')
            raise

        print("\nUso fallback manuale controllato.")
        next_cfg = make_manual_fallback(current, args.goal)

    next_cfg["note"] = "Generated by UPU LLM controller v2. Review before replacing base config."
    save_json(args.out_config, next_cfg)

    print("\n=== Saved next config ===")
    print(args.out_config)
    print(json.dumps(next_cfg, indent=2, ensure_ascii=False))

    if args.render:
        render_html(args.render_script, args.out_config, args.html)

if __name__ == "__main__":
    main()
