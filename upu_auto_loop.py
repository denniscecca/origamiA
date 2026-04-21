#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UPU Auto Loop — LLM lento + HTML vivo.

Ciclo:
1. legge config corrente
2. genera scena HTML interattiva live_scene.html
3. calcola metriche Φ, K, T, E_C, η, mixed_alive, ustar_return
4. calcola reward
5. chiede al LLM locale una micro-variazione JSON
6. salva storia, best config e aggiorna live_status.json
7. ripete

Richiede:
- plotly installato
- upu_origami_interactive_4d_config.py nella stessa cartella
- llama.cpp server acceso su http://127.0.0.1:8080/v1/chat/completions

Uso:
python3 upu_auto_loop.py --cycles 100 --sleep 2 --render --frames 50 

In un altro terminale, per vedere "live":
python3 -m http.server 7777
Poi apri:
http://127.0.0.1:7777/upu_live_viewer.html
"""

import argparse
import itertools
import json
import math
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

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

DEFAULT = {
    "phi_gain": 1.0,
    "k_gain": 1.0,
    "temperature": 0.50,
    "ustar_pull": 0.18,
    "four_u_radius": 1.80,
    "golden_lock": 1.61803398875,
    "z_coherence_gain": 1.0,
    "mixed_bridge_gain": 0.72,
    "rotation_speed": 1.0,
    "phase_twist": 0.55,
}

def clamp(k, v):
    lo, hi = ALLOWED[k]
    return max(lo, min(hi, float(v)))

def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return dict(default or DEFAULT)
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def states():
    return list(itertools.product([-1, 1], repeat=4))

def classify(s):
    if s in [(1,1,1,1), (-1,-1,-1,-1)]:
        return "coherent"
    if s in {(1,-1,1,-1), (-1,1,-1,1), (1,-1,-1,1), (-1,1,1,-1)}:
        return "alternating"
    return "mixed"

def rot4(i, j, a):
    R = np.eye(4)
    c, ss = math.cos(a), math.sin(a)
    R[i,i] = c; R[j,j] = c
    R[i,j] = -ss; R[j,i] = ss
    return R

def project(s, t, cfg, radius=2.8):
    phi = float(cfg.get("golden_lock", 1.61803398875))
    speed = float(cfg.get("rotation_speed", 1.0))
    twist = float(cfg.get("phase_twist", 0.55))
    a = 2 * math.pi * t * speed

    R = (
        rot4(0, 1, 0.80 * a)
        @ rot4(2, 3, twist * a * phi)
        @ rot4(0, 2, 0.32 * a)
        @ rot4(1, 3, 0.21 * a * phi)
    )

    P = np.array([
        [0.75 * cfg["k_gain"], -0.75 * cfg["k_gain"], 0.20, -0.20],
        [-0.20, 0.20, 0.75 * cfg["k_gain"], -0.75 * cfg["k_gain"]],
        [0.60 * cfg["z_coherence_gain"], 0.60 * cfg["z_coherence_gain"],
         0.60 * cfg["z_coherence_gain"], 0.60 * cfg["z_coherence_gain"]],
    ])

    p = P @ R @ np.array(s, dtype=float)
    n = np.linalg.norm(p) + 1e-12
    p = p / max(1.0, n / radius)

    c = classify(s)
    if c == "coherent":
        p[0] *= 0.35
        p[1] *= 0.35
        p[2] *= 1.05 * cfg["phi_gain"]
    elif c == "alternating":
        rxy = math.sqrt(p[0]**2 + p[1]**2) + 1e-12
        target = cfg["four_u_radius"]
        p[0] = p[0] / rxy * target
        p[1] = p[1] / rxy * target
        p[2] *= 0.25 + 0.15 * cfg["temperature"]
    else:
        p *= cfg["mixed_bridge_gain"] * (0.85 + 0.30 * cfg["temperature"])

    pull = max(0.0, min(0.95, cfg["ustar_pull"]))
    if c == "mixed":
        p *= (1.0 - 0.35 * pull)
    elif c == "alternating":
        p *= (1.0 - 0.15 * pull)

    return p

def compute_metrics(cfg, samples=24):
    all_states = states()
    vals = []
    for k in range(samples):
        t = k / max(1, samples - 1)
        coords = {s: project(s, t, cfg) for s in all_states}
        coh = [coords[s] for s in all_states if classify(s) == "coherent"]
        alt = [coords[s] for s in all_states if classify(s) == "alternating"]
        mix = [coords[s] for s in all_states if classify(s) == "mixed"]

        axis = np.linalg.norm(coh[0] - coh[1]) if len(coh) == 2 else 0.0
        alt_r = float(np.mean([np.linalg.norm(p[:2]) for p in alt])) if alt else 0.0
        mix_r = float(np.mean([np.linalg.norm(p) for p in mix])) if mix else 0.0

        Phi = float(np.clip(axis / 5.6, 0, 1))
        K = float(np.clip(alt_r / 1.8, 0, 1))
        T = float(np.clip(mix_r / 2.8, 0, 1))
        E_C = Phi * Phi * K
        eta = E_C / (E_C + T + 1e-9)

        # mixed_alive: non deve collassare a 0, ma nemmeno dominare.
        mixed_target = 0.36
        mixed_alive = float(np.clip(1.0 - abs(T - mixed_target) / mixed_target, 0, 1))

        # uStar return: pull alto + distanza media non troppo esterna.
        mean_r = float(np.mean([np.linalg.norm(p) for p in coords.values()]))
        ustar_return = float(np.clip(1.0 - mean_r / 3.0, 0, 1))
        vals.append((Phi, K, T, E_C, eta, mixed_alive, ustar_return))

    arr = np.array(vals)
    names = ["Phi", "K", "T", "E_C", "eta", "mixed_alive", "ustar_return"]
    out = {name: float(arr[:, i].mean()) for i, name in enumerate(names)}

    # reward: massimizza eta e uStar, ma conserva mixed_alive e K.
    out["reward"] = float(
        0.34 * out["eta"] +
        0.26 * out["ustar_return"] +
        0.22 * out["mixed_alive"] +
        0.12 * out["K"] +
        0.06 * out["Phi"]
    )
    return out

def extract_json(text):
    text = text.strip()
    text = text.replace("```json", "```").replace("```JSON", "```")
    if text.startswith("```"):
        text = text.strip("`").strip()
    a = text.find("{")
    b = text.rfind("}")
    if a == -1 or b == -1 or b <= a:
        raise ValueError("No JSON object found")
    return json.loads(text[a:b+1])

def propose_with_llm(api, model, cfg, metrics, goal, max_tokens=350):
    system = (
        "You are a numeric controller for a UPU Origami 4D field. "
        "Return ONLY valid JSON, no markdown, no prose. "
        "Your response must start with { and end with }. "
        "Make small changes, normally within +/-0.05 or +/-0.10. "
        "Do not collapse mixed_bridge_gain. Keep golden_lock close to 1.61803398875."
    )
    user = {
        "goal": goal,
        "current_config": {k: cfg[k] for k in ALLOWED},
        "metrics": metrics,
        "allowed_ranges": ALLOWED,
        "reward_logic": "maximize reward, eta, ustar_return; keep mixed_alive nonzero; avoid high T collapse",
        "return_only_keys": list(ALLOWED.keys()),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
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
    with urllib.request.urlopen(req, timeout=240) as r:
        data = json.loads(r.read().decode("utf-8"))
    raw = data["choices"][0]["message"]["content"]
    proposed = extract_json(raw)
    nxt = dict(cfg)
    for k in ALLOWED:
        if k in proposed:
            nxt[k] = clamp(k, proposed[k])
    return nxt, raw

def propose_fallback(cfg, metrics):
    # Piccolo regolatore ECR euristico se il modello parla invece di JSON.
    nxt = dict(cfg)
    if metrics["T"] > 0.40:
        nxt["temperature"] = clamp("temperature", nxt["temperature"] - 0.04)
        nxt["ustar_pull"] = clamp("ustar_pull", nxt["ustar_pull"] + 0.03)
    elif metrics["mixed_alive"] < 0.45:
        nxt["mixed_bridge_gain"] = clamp("mixed_bridge_gain", nxt["mixed_bridge_gain"] + 0.03)
        nxt["temperature"] = clamp("temperature", nxt["temperature"] + 0.02)
    else:
        nxt["ustar_pull"] = clamp("ustar_pull", nxt["ustar_pull"] + 0.02)
        nxt["phi_gain"] = clamp("phi_gain", nxt["phi_gain"] + 0.02)

    # micro jitter sul phase_twist per non restare sterile
    nxt["phase_twist"] = clamp("phase_twist", nxt["phase_twist"] + random.choice([-0.02, 0.02]))
    return nxt

def render_scene(render_script, cfg_path, html_path, frames, labels=False):
    cmd = [
        sys.executable,
        str(render_script),
        "--config", str(cfg_path),
        "--out", str(html_path),
        "--frames", str(frames),
    ]
    if labels:
        cmd.append("--labels")
    subprocess.check_call(cmd)

def write_status(cycle, cfg, metrics, best_reward, version):
    status = {
        "cycle": cycle,
        "version": version,
        "reward": metrics["reward"],
        "best_reward": best_reward,
        "metrics": {k: metrics[k] for k in ["Phi", "K", "T", "E_C", "eta", "mixed_alive", "ustar_return"]},
        "config": {k: cfg[k] for k in ALLOWED},
        "updated_at": time.time(),
    }
    save_json("live_status.json", status)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="local")
    ap.add_argument("--config", default="./upu_field_config.json")
    ap.add_argument("--render-script", default="./upu_origami_interactive_4d_config.py")
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="usa solo fallback euristico, senza chiamare LLM")
    ap.add_argument("--goal", default="maximize eta and uStar return while keeping mixed states alive; do not collapse 4U ring")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_json(cfg_path, DEFAULT)
    for k, v in DEFAULT.items():
        cfg.setdefault(k, v)
    for k in ALLOWED:
        cfg[k] = clamp(k, cfg[k])

    render_script = Path(args.render_script)
    history_dir = Path("upu_loop_history")
    history_dir.mkdir(exist_ok=True)

    # viewer file locale
    viewer = Path("upu_live_viewer.html")
    if not viewer.exists():
        viewer.write_text(LIVE_VIEWER_HTML, encoding="utf-8")

    best_reward = -1.0
    best_cfg = dict(cfg)
    version = 0

    for cycle in range(args.cycles):
        metrics = compute_metrics(cfg)
        reward = metrics["reward"]

        if reward > best_reward:
            best_reward = reward
            best_cfg = dict(cfg)
            save_json("best_config.json", best_cfg)

        # salva config corrente e status prima del render
        save_json("live_config.json", cfg)
        save_json(history_dir / f"config_{cycle:04d}.json", cfg)
        save_json(history_dir / f"metrics_{cycle:04d}.json", metrics)
        write_status(cycle, cfg, metrics, best_reward, version)

        print(f"[cycle {cycle:03d}] reward={reward:.4f} best={best_reward:.4f} "
              f"Phi={metrics['Phi']:.3f} K={metrics['K']:.3f} T={metrics['T']:.3f} "
              f"eta={metrics['eta']:.3f} mixed={metrics['mixed_alive']:.3f} uStar={metrics['ustar_return']:.3f}")

        if args.render:
            try:
                render_scene(render_script, "live_config.json", "live_scene.html", args.frames, args.labels)
                shutil.copyfile("live_scene.html", history_dir / f"scene_{cycle:04d}.html")
                version += 1
                write_status(cycle, cfg, metrics, best_reward, version)
            except Exception as e:
                print(f"[render error] {e}")

        # proposta prossimo passo
        if args.no_llm:
            next_cfg = propose_fallback(cfg, metrics)
            raw = "NO_LLM fallback"
        else:
            try:
                next_cfg, raw = propose_with_llm(args.api, args.model, cfg, metrics, args.goal)
            except Exception as e:
                print(f"[llm/json error] {e}")
                next_cfg = propose_fallback(cfg, metrics)
                raw = f"fallback because: {e}"

        Path(history_dir / f"llm_raw_{cycle:04d}.txt").write_text(str(raw), encoding="utf-8")
        cfg = next_cfg
        save_json(cfg_path, cfg)

        if args.sleep > 0:
            time.sleep(args.sleep)

    # render finale della migliore
    save_json("best_config.json", best_cfg)
    print(f"\nDONE. best_reward={best_reward:.4f}")
    print("Open viewer: http://127.0.0.1:7777/upu_live_viewer.html")
    print("Or open file: upu_live_viewer.html")

LIVE_VIEWER_HTML = '<!doctype html>\n<html lang="it">\n<head>\n  <meta charset="utf-8" />\n  <title>UPU Live Field Viewer</title>\n  <meta name="viewport" content="width=device-width, initial-scale=1" />\n  <style>\n    body { margin: 0; background: #08090d; color: #e9edf7; font-family: system-ui, sans-serif; }\n    header { padding: 10px 14px; border-bottom: 1px solid #242938; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }\n    .pill { background: #151927; border: 1px solid #30384f; border-radius: 999px; padding: 4px 10px; font-size: 13px; }\n    iframe { width: 100vw; height: calc(100vh - 58px); border: none; background: #fff; }\n    button { background: #24314f; color: white; border: 1px solid #546898; padding: 6px 10px; border-radius: 10px; }\n  </style>\n</head>\n<body>\n<header>\n  <strong>UPU Live Field</strong>\n  <span class="pill" id="cycle">cycle: ...</span>\n  <span class="pill" id="reward">reward: ...</span>\n  <span class="pill" id="metrics">Φ/K/T/E/η: ...</span>\n  <button onclick="reloadScene()">reload now</button>\n</header>\n<iframe id="scene" src="./live_scene.html"></iframe>\n\n<script>\nlet lastVersion = null;\n\nasync function fetchJSON(path) {\n  const r = await fetch(path + "?t=" + Date.now());\n  if (!r.ok) throw new Error("missing " + path);\n  return await r.json();\n}\n\nfunction reloadScene() {\n  const frame = document.getElementById("scene");\n  frame.src = "./live_scene.html?t=" + Date.now();\n}\n\nasync function tick() {\n  try {\n    const status = await fetchJSON("./live_status.json");\n    document.getElementById("cycle").textContent = "cycle: " + status.cycle;\n    document.getElementById("reward").textContent = "reward: " + Number(status.reward).toFixed(4);\n    document.getElementById("metrics").textContent =\n      "Φ=" + Number(status.metrics.Phi).toFixed(3) +\n      " K=" + Number(status.metrics.K).toFixed(3) +\n      " T=" + Number(status.metrics.T).toFixed(3) +\n      " E=" + Number(status.metrics.E_C).toFixed(3) +\n      " η=" + Number(status.metrics.eta).toFixed(3);\n\n    if (lastVersion !== status.version) {\n      lastVersion = status.version;\n      reloadScene();\n    }\n  } catch(e) {\n    document.getElementById("cycle").textContent = "waiting for loop...";\n  }\n}\n\nsetInterval(tick, 2500);\ntick();\n</script>\n</body>\n</html>\n'

if __name__ == "__main__":
    main()
