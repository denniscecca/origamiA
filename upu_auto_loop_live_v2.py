#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UPU Auto Loop LIVE v2 — loop continuo non bloccante.

Differenza dalla v1:
- il campo continua SEMPRE a girare
- l'LLM viene chiamato solo ogni --llm-every cicli
- la chiamata LLM gira in background thread
- se l'LLM va in palla, il loop usa fallback ECR e non si ferma
- il viewer live continua ad aggiornarsi

Richiede nella stessa cartella:
- upu_auto_loop.py
- upu_origami_interactive_4d_config.py
- upu_field_config.json
- upu_live_viewer.html

Uso consigliato:
python3 upu_auto_loop_live_v2.py \
  --cycles 999999 \
  --sleep 0.7 \
  --render \
  --frames 999999 \
  --llm-every 8 \
  --llm-timeout 45
"""

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Importiamo la logica già definita nella v1.
import upu_auto_loop as base


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path, default):
    p = Path(path)
    if not p.exists():
        return dict(default)
    return json.loads(p.read_text(encoding="utf-8"))


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


def llm_job(api, model, cfg, metrics, goal, max_tokens):
    """Eseguito in background. Può fallire senza fermare il campo."""
    return base.propose_with_llm(api, model, cfg, metrics, goal, max_tokens=max_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="local")
    ap.add_argument("--config", default="./upu_field_config.json")
    ap.add_argument("--render-script", default="./upu_origami_interactive_4d_config.py")
    ap.add_argument("--cycles", type=int, default=999999)
    ap.add_argument("--sleep", type=float, default=0.7)
    ap.add_argument("--frames", type=int, default=18)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--llm-every", type=int, default=8, help="chiama LLM ogni N cicli")
    ap.add_argument("--llm-timeout", type=float, default=45.0, help="dopo questi secondi ignora la risposta LLM pendente")
    ap.add_argument("--llm-max-tokens", type=int, default=220)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-history-scenes", action="store_true", help="non copiare ogni HTML nella history, risparmia disco")
    ap.add_argument("--goal", default="Return only JSON. Increase eta and uStar return while keeping mixed states alive. Do not explain.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_json(cfg_path, base.DEFAULT)
    for k, v in base.DEFAULT.items():
        cfg.setdefault(k, v)
    for k in base.ALLOWED:
        cfg[k] = base.clamp(k, cfg[k])

    history_dir = Path("upu_loop_history")
    history_dir.mkdir(exist_ok=True)

    viewer = Path("upu_live_viewer.html")
    if not viewer.exists():
        viewer.write_text(base.LIVE_VIEWER_HTML, encoding="utf-8")

    best_reward = -1.0
    best_cfg = dict(cfg)
    version = 0

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    llm_future = None
    llm_started_at = None
    pending_cycle = None

    print("[live-v2] campo continuo avviato")
    print("[live-v2] LLM non blocca il loop; se tarda, il campo continua con fallback ECR")

    try:
        for cycle in range(args.cycles):
            t0 = time.time()

            # 1) metriche config corrente
            metrics = base.compute_metrics(cfg)
            reward = metrics["reward"]

            if reward > best_reward:
                best_reward = reward
                best_cfg = dict(cfg)
                save_json("best_config.json", best_cfg)

            # 2) status live
            save_json("live_config.json", cfg)
            save_json(history_dir / f"config_{cycle:04d}.json", cfg)
            save_json(history_dir / f"metrics_{cycle:04d}.json", metrics)
            base.write_status(cycle, cfg, metrics, best_reward, version)

            print(
                f"[cycle {cycle:04d}] reward={reward:.4f} best={best_reward:.4f} "
                f"Phi={metrics['Phi']:.3f} K={metrics['K']:.3f} T={metrics['T']:.3f} "
                f"eta={metrics['eta']:.3f} mixed={metrics['mixed_alive']:.3f} "
                f"uStar={metrics['ustar_return']:.3f}"
            )

            # 3) render veloce, opzionale
            if args.render:
                try:
                    render_scene(args.render_script, "live_config.json", "live_scene.html", args.frames, args.labels)
                    if not args.no_history_scenes:
                        shutil.copyfile("live_scene.html", history_dir / f"scene_{cycle:04d}.html")
                    version += 1
                    base.write_status(cycle, cfg, metrics, best_reward, version)
                except Exception as e:
                    print(f"[render error] {e}")

            # 4) se c'è una risposta LLM pronta, applicala
            applied_llm = False
            if llm_future is not None:
                if llm_future.done():
                    try:
                        llm_cfg, raw = llm_future.result()
                        Path(history_dir / f"llm_raw_{pending_cycle:04d}.txt").write_text(str(raw), encoding="utf-8")
                        cfg = llm_cfg
                        print(f"[cycle {cycle:04d}] applied LLM config from cycle {pending_cycle}")
                        applied_llm = True
                    except Exception as e:
                        print(f"[cycle {cycle:04d}] LLM failed -> fallback continues: {e}")
                    llm_future = None
                    llm_started_at = None
                    pending_cycle = None
                elif llm_started_at and (time.time() - llm_started_at) > args.llm_timeout:
                    print(f"[cycle {cycle:04d}] LLM timeout -> ignored pending request, fallback continues")
                    llm_future = None
                    llm_started_at = None
                    pending_cycle = None

            # 5) se non abbiamo appena applicato LLM, fai sempre un micro-step fallback
            # così il campo non resta fermo mai.
            if not applied_llm:
                cfg = base.propose_fallback(cfg, metrics)

            # 6) avvia una nuova chiamata LLM solo ogni N cicli, se non ce n'è già una
            if (not args.no_llm) and (llm_future is None) and (cycle % max(1, args.llm_every) == 0):
                try:
                    frozen_cfg = dict(cfg)
                    frozen_metrics = dict(metrics)
                    llm_future = executor.submit(
                        llm_job,
                        args.api,
                        args.model,
                        frozen_cfg,
                        frozen_metrics,
                        args.goal,
                        args.llm_max_tokens,
                    )
                    llm_started_at = time.time()
                    pending_cycle = cycle
                    print(f"[cycle {cycle:04d}] LLM background request started")
                except Exception as e:
                    print(f"[cycle {cycle:04d}] cannot start LLM request: {e}")
                    llm_future = None
                    llm_started_at = None
                    pending_cycle = None

            # 7) salva config per ciclo successivo
            save_json(cfg_path, cfg)

            # 8) sleep regolare, compensando il tempo già usato
            elapsed = time.time() - t0
            wait = max(0.0, args.sleep - elapsed)
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        print("\n[live-v2] stop manuale ricevuto")

    finally:
        save_json("best_config.json", best_cfg)
        executor.shutdown(wait=False, cancel_futures=True)
        print(f"[live-v2] DONE. best_reward={best_reward:.4f}")
        print("[live-v2] viewer: http://127.0.0.1:7777/upu_live_viewer.html")


if __name__ == "__main__":
    main()
