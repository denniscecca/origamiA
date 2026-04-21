#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UPU Origami — interactive 4D→3D projection, config-aware.

Genera HTML interattivo ruotabile con mouse/touch.
Legge un JSON di controllo modificabile da LLM/Msty.

Uso:
python3 upu_origami_interactive_4d_config.py --config upu_field_config.json --out upu_scene.html
"""

import argparse, itertools, json, math
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

DEFAULT = {
    "phi_gain": 1.0,
    "k_gain": 1.0,
    "temperature": 0.50,
    "ustar_pull": 0.18,
    "four_u_radius": 1.80,
    "golden_lock": (1 + 5 ** 0.5) / 2,
    "z_coherence_gain": 1.0,
    "mixed_bridge_gain": 0.72,
    "rotation_speed": 1.0,
    "phase_twist": 0.55,
}

def load_cfg(path):
    cfg = dict(DEFAULT)
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        for k, v in user.items():
            if k in cfg:
                try:
                    cfg[k] = float(v)
                except Exception:
                    pass
    return cfg

def states():
    return list(itertools.product([-1, 1], repeat=4))

def label(s):
    return "".join("+" if x == 1 else "-" for x in s)

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def cls(s):
    if s in [(1,1,1,1), (-1,-1,-1,-1)]:
        return "coherent"
    if s in {(1,-1,1,-1), (-1,1,-1,1), (1,-1,-1,1), (-1,1,1,-1)}:
        return "alternating"
    return "mixed"

def rot4(i, j, a):
    R = np.eye(4)
    c, s = math.cos(a), math.sin(a)
    R[i,i] = c; R[j,j] = c
    R[i,j] = -s; R[j,i] = s
    return R

def project(s, t, cfg, radius=2.8):
    phi = cfg["golden_lock"]
    speed = cfg["rotation_speed"]
    twist = cfg["phase_twist"]
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

    c = cls(s)
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

    # uStar pull: ritorno dolce verso centro, non collasso totale
    pull = max(0.0, min(0.95, cfg["ustar_pull"]))
    if c == "mixed":
        p *= (1.0 - 0.35 * pull)
    elif c == "alternating":
        p *= (1.0 - 0.15 * pull)

    return p

def metric(all_states, coords):
    coh = [coords[s] for s in all_states if cls(s) == "coherent"]
    alt = [coords[s] for s in all_states if cls(s) == "alternating"]
    mix = [coords[s] for s in all_states if cls(s) == "mixed"]
    axis = np.linalg.norm(coh[0] - coh[1]) if len(coh) == 2 else 0
    alt_r = float(np.mean([np.linalg.norm(p[:2]) for p in alt])) if alt else 0
    mix_r = float(np.mean([np.linalg.norm(p) for p in mix])) if mix else 0
    Phi = float(np.clip(axis / 5.6, 0, 1))
    K = float(np.clip(alt_r / 1.8, 0, 1))
    T = float(np.clip(mix_r / 2.8, 0, 1))
    EC = Phi * Phi * K
    eta = EC / (EC + T + 1e-9)
    return Phi, K, T, EC, eta

def sphere(radius=2.8):
    traces = []
    for ph in np.linspace(0, 2*np.pi, 18):
        th = np.linspace(0, np.pi, 80)
        traces.append(go.Scatter3d(
            x=radius*np.cos(ph)*np.sin(th),
            y=radius*np.sin(ph)*np.sin(th),
            z=radius*np.cos(th),
            mode="lines", line=dict(width=1, color="rgba(120,170,220,0.20)"),
            hoverinfo="skip", showlegend=False))
    for th in np.linspace(0.2, np.pi-0.2, 10):
        ph = np.linspace(0, 2*np.pi, 100)
        traces.append(go.Scatter3d(
            x=radius*np.cos(ph)*np.sin(th),
            y=radius*np.sin(ph)*np.sin(th),
            z=radius*np.cos(th)*np.ones_like(ph),
            mode="lines", line=dict(width=1, color="rgba(120,170,220,0.15)"),
            hoverinfo="skip", showlegend=False))
    return traces

def static_geometry():
    traces = [go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers+text",
        marker=dict(size=8, color="gold", symbol="diamond"),
        text=["uStar<br>Unity"], textposition="top center", name="uStar")]
    base = np.array([[1.8,0,0], [0,1.8,0], [-1.8,0,0], [0,-1.8,0]])
    north = np.array([0,0,2.45]); south = np.array([0,0,-2.45])
    lines = []
    for i in range(4):
        lines.append((base[i], base[(i+1)%4]))
        lines.append((north, base[i]))
        lines.append((south, base[i]))
    for a,b in lines:
        traces.append(go.Scatter3d(
            x=[a[0],b[0]], y=[a[1],b[1]], z=[a[2],b[2]],
            mode="lines", line=dict(width=3, color="rgba(80,80,80,0.25)"),
            hoverinfo="skip", showlegend=False))
    return traces

def frame_data(all_states, t, cfg, labels=False):
    coords = {s: project(s, t, cfg) for s in all_states}
    data = []

    # edges
    ex=[]; ey=[]; ez=[]
    for i,a in enumerate(all_states):
        for b in all_states[i+1:]:
            if hamming(a,b) == 1:
                pa,pb = coords[a], coords[b]
                ex += [pa[0], pb[0], None]
                ey += [pa[1], pb[1], None]
                ez += [pa[2], pb[2], None]
    data.append(go.Scatter3d(x=ex,y=ey,z=ez, mode="lines",
        line=dict(width=2, color="rgba(80,80,80,0.20)"),
        hoverinfo="skip", name="4D hypercube edges"))

    # 16V rays
    vx=[]; vy=[]; vz=[]
    for s in all_states:
        p = coords[s]; n = np.linalg.norm(p)+1e-12; o = p/n*2.75
        vx += [o[0],0,None]; vy += [o[1],0,None]; vz += [o[2],0,None]
    data.append(go.Scatter3d(x=vx,y=vy,z=vz, mode="lines",
        line=dict(width=2, color="rgba(255,180,60,0.22)"),
        hoverinfo="skip", name="16V rays"))

    styles = {
        "coherent": ("red", 7, "circle", "coherent / c-Si"),
        "alternating": ("blue", 7, "square", "alternating / 4U"),
        "mixed": ("gray", 5, "circle", "mixed bridge"),
    }
    groups = {k: [s for s in all_states if cls(s)==k] for k in styles}
    for k, group in groups.items():
        pts = np.array([coords[s] for s in group])
        color, size, symbol, name = styles[k]
        txt = [label(s) for s in group]
        data.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode="markers+text" if labels else "markers",
            marker=dict(size=size, color=color, symbol=symbol, line=dict(width=1, color="black")),
            text=txt if labels else None,
            customdata=txt,
            hovertemplate="state=%{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            name=name))

    # coherent axis
    coh = groups["coherent"]
    a,b = coords[coh[0]], coords[coh[1]]
    data.append(go.Scatter3d(x=[a[0],b[0]], y=[a[1],b[1]], z=[a[2],b[2]],
        mode="lines", line=dict(width=8, color="rgba(220,0,0,0.65)"),
        hoverinfo="skip", name="coherent axis"))

    # 4U ring
    alt = groups["alternating"]
    center = np.mean([coords[s] for s in alt], axis=0)
    alt = sorted(alt, key=lambda s: math.atan2(coords[s][1]-center[1], coords[s][0]-center[0]))
    rx=[]; ry=[]; rz=[]
    for i in range(len(alt)+1):
        p = coords[alt[i % len(alt)]]
        rx.append(p[0]); ry.append(p[1]); rz.append(p[2])
    data.append(go.Scatter3d(x=rx,y=ry,z=rz, mode="lines",
        line=dict(width=6, color="rgba(0,90,220,0.70)"),
        hoverinfo="skip", name="4U alternating ring"))

    Phi,K,T,EC,eta = metric(all_states, coords)
    title = f"UPU 4D→3D config | t={t:.3f} | Φ={Phi:.3f} K={K:.3f} T={T:.3f} E_C={EC:.3f} η={eta:.3f}"
    return data, title

def build(out, cfg, frames_count=80, labels=False):
    all_states = states()
    static = sphere() + static_geometry()
    first, title = frame_data(all_states, 0.0, cfg, labels)
    fig = go.Figure(data=static + first)
    n_static = len(static)

    frames = []
    for k in range(frames_count):
        t = k / max(1, frames_count-1)
        data, title_k = frame_data(all_states, t, cfg, labels)
        frames.append(go.Frame(
            data=data,
            traces=list(range(n_static, n_static + len(data))),
            name=str(k),
            layout=go.Layout(title=title_k)))
    fig.frames = frames

    fig.update_layout(
        title=title,
        width=1100, height=820,
        scene=dict(
            xaxis_title="X / 4U phase", yaxis_title="Y / phase membrane", zaxis_title="Z / coherent pass",
            xaxis=dict(range=[-3.2,3.2]), yaxis=dict(range=[-3.2,3.2]), zaxis=dict(range=[-3.2,3.2]),
            aspectmode="cube"),
        margin=dict(l=0,r=0,t=60,b=0),
        legend=dict(x=0.01,y=0.98),
        sliders=[{
            "active": 0, "currentvalue": {"prefix": "frame: "},
            "steps": [{
                "label": str(k), "method": "animate",
                "args": [[str(k)], {"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}]
            } for k in range(frames_count)]
        }],
        updatemenus=[{
            "type": "buttons", "showactive": False, "x": 0.02, "y": 0.08,
            "buttons": [
                {"label":"Play", "method":"animate",
                 "args":[None, {"frame":{"duration":80,"redraw":True}, "fromcurrent":True, "transition":{"duration":0}}]},
                {"label":"Pause", "method":"animate",
                 "args":[[None], {"frame":{"duration":0,"redraw":False}, "mode":"immediate", "transition":{"duration":0}}]},
            ]}]
    )
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    print(f"Saved: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./upu_field_config.json")
    ap.add_argument("--out", default="./upu_scene.html")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--labels", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    build(Path(args.out), cfg, args.frames, args.labels)

if __name__ == "__main__":
    main()
