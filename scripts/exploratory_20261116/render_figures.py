#!/usr/bin/env python3
"""evidence_tier: exploratory_numerical_sandbox

Render standalone scientific SVG figures from the two existing probe summaries.
Only the Python standard library is used. No experiments are run or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/"results/exploratory_20261116"
TIER = "exploratory_numerical_sandbox"
FONT = "Arial, Helvetica, sans-serif"


def esc(value):
    return html.escape(str(value), quote=True)


def interpolate(stops, position):
    position = max(0., min(1., position))*(len(stops)-1)
    i = min(int(position), len(stops)-2)
    u = position-i
    colors = [[int(c[k:k+2], 16) for k in (1, 3, 5)] for c in stops[i:i+2]]
    return "#"+"".join(f"{round(a+(b-a)*u):02x}" for a, b in zip(*colors))


def risk_position(value):
    return math.log1p(max(0., value)/.001)/math.log1p(.5/.001)


def risk_color(value):
    return interpolate(["#fff7ec", "#fee8c8", "#fdbb84", "#fc8d59", "#d7301f", "#7f0000"], risk_position(value))


def angle_color(value):
    return interpolate(["#f7fcfd", "#ccece6", "#66c2a4", "#238b45", "#00441b"], value)


class SVG:
    def __init__(self, width, height, title, subtitle, seed):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      f'<title>{esc(title)}</title>',
                      f'<desc>{esc("evidence_tier: "+TIER+"; seed="+str(seed)+"; paired descriptive numerical sandbox; no neural-network experiment")}</desc>',
                      f'<style>text{{font-family:{FONT}}} .muted{{fill:#586574}}</style>',
                      '<rect width="100%" height="100%" fill="white"/>']
        self.text(32, 34, title, 24, weight="bold")
        self.text(32, 58, subtitle, 13)
        self.text(32, 79, f"evidence_tier: {TIER} | seed: {seed} | paired descriptive", 11, color="#586574")

    def text(self, x, y, value, size=12, anchor="start", color="#17212b", weight="normal", rotate=None):
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}"{transform}>{esc(value)}</text>')

    def line(self, x1, y1, x2, y2, color="#8b99a6", width=1, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def rect(self, x, y, w, h, fill, stroke="#ffffff", sw=1):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def circle(self, x, y, r, fill, stroke=None):
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{stroke or fill}"/>')

    def triangle(self, x, y, color="#111111"):
        self.parts.append(f'<path d="M{x-4:.2f},{y+3:.2f} L{x+4:.2f},{y+3:.2f} L{x:.2f},{y-4:.2f} Z" fill="{color}" stroke="white" stroke-width=".7"/>')

    def path(self, points, color, width=2, dash=None):
        coords = " ".join(f'{x:.2f},{y:.2f}' for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

    def finish(self):
        return "\n".join(self.parts+["</svg>"])+"\n"


def legend(svg, x, y, width, kind="risk"):
    for i in range(160):
        t = i/159
        v = .001*math.expm1(t*math.log1p(.5/.001)) if kind == "risk" else t
        svg.rect(x+i*width/160, y, width/160+.2, 16, risk_color(v) if kind == "risk" else angle_color(v), sw=0)
    ticks = [0, .001, .01, .05, .1, .25, .5] if kind == "risk" else [0, .2, .4, .6, .8, 1]
    for v in ticks:
        t = risk_position(v) if kind == "risk" else v
        svg.line(x+width*t, y+16, x+width*t, y+21, color="#17212b")
        svg.text(x+width*t, y+34, f"{100*v:g}%" if kind == "risk" else f"{v:g}", 11, "middle")
    svg.text(x, y-8, "Observed error rate (log1p color scale; saturated at 50%)" if kind == "risk" else "Mean sin(theta_max), linear color scale", 12)


def phase_figure(data, mode):
    phase = data["phase"]
    times = phase["grid"]["T"]
    r2s = list(reversed(phase["grid"]["r2_population"]))
    alphas = phase["grid"]["alphas"]
    processes = list(phase["grid"]["processes"])
    title = "Finite-window physical slow-set risk: "+("all answers" if mode == "all" else "own-CAL median-gap selection")
    svg = SVG(1600, 830, title,
              "Truth: same-window physical top-2; predicted scores / independent REF variance (ddof=0); lag = {4,16,32}.", data["seed"])
    lookup = {(r["process"], r["alpha"], r["T"], r["r2_population"]): r for r in phase["cells"]}
    cw, ch, pw, ph = 25, 22, 310, 285
    for ip, process in enumerate(processes):
        for ia, alpha in enumerate(alphas):
            ox, oy = 72+ia*pw, 146+ip*ph
            svg.text(ox+125, oy-33, process.replace("separated_", "")+f" | alpha={alpha:g}", 15, "middle", weight="bold")
            svg.text(ox+125, oy-14, "population state R2", 10, "middle", color="#586574")
            for iy, r2 in enumerate(r2s):
                svg.text(ox-7, oy+iy*ch+15, f"{r2:g}", 10, "end")
                for ix, t in enumerate(times):
                    row = lookup[process, alpha, t, r2]
                    value = row[mode]["risk"]
                    x, y = ox+ix*cw, oy+iy*ch
                    svg.rect(x, y, cw, ch, risk_color(value), sw=.6)
                    if value < .01:
                        svg.circle(x+cw/2, y+ch/2, 2.2, "#202b36", "white")
            for ix, t in enumerate(times):
                svg.text(ox+ix*cw+cw/2, oy+len(r2s)*ch+13, t, 9, "end", rotate=-50)
            svg.text(ox+125, oy+len(r2s)*ch+59, "Window length T (states)", 11, "middle")
    legend(svg, 180, 729, 800)
    svg.circle(1040, 738, 3, "#202b36", "white")
    svg.text(1053, 742, "Observed risk < 1%", 12)
    svg.text(32, 798, "CAL=2048 independent windows; TEST=4096; REF=32768 states. Dots are point estimates, not simultaneous risk certificates.", 12)
    svg.text(32, 817, "Nested T prefixes and shared innovations across alpha/R2; cells are paired. own-CAL coverage is approximately 50%, not fixed at exactly 50%.", 11, color="#586574")
    return svg.finish()


def subspace_figure(data):
    phase = data["subspace_phase"]
    powers = phase["design"]["signal_power_parameters"]
    eps = sorted([round(1-p, 6) for p in powers], reverse=True)
    gaps = phase["design"]["boundary_rho_gaps"]
    lags = phase["design"]["calibration_lags"]
    residuals = ["isotropic_ar05", "slow_noise_on_fast_axes"]
    svg = SVG(1480, 875, "Unlabeled slow-subspace recovery: sampling noise and residual spectral bias",
              "Ncal=1024 independent trajectories x 193 states; 3 CAL replicates; color = mean signal-loading subspace angle.", data["seed"])
    lookup = {(r["residual"], r["cal_lag"], round(1-r["signal_power_parameter"], 6), r["gap_rho"]): r
              for r in phase["cell_means"] if r["N_cal_trajectories"] == 1024}
    cw, ch = 70, 30
    for ir, residual in enumerate(residuals):
        for il, lag in enumerate(lags):
            ox, oy = 100+il*475, 143+ir*300
            label = "Isotropic AR(.5) residual" if ir == 0 else "Slow AR(.94) noise on fast axes"
            svg.text(ox+175, oy-22, label+f" | lag={lag}", 14, "middle", weight="bold")
            for iy, e in enumerate(eps):
                svg.text(ox-9, oy+iy*ch+20, f"{e:g}", 11, "end")
                for ix, gap in enumerate(gaps):
                    r = lookup[residual, lag, e, gap]
                    value = r["signal_loading_subspace_sin_theta_max_mean"]
                    x, y = ox+ix*cw, oy+iy*ch
                    svg.rect(x, y, cw, ch, angle_color(value), sw=.7)
                    svg.text(x+cw/2, y+20, f"{value:.3f}", 12, "middle", color="white" if value>.65 else "#17212b")
                    if not r["population_slow_subspace_is_target"]:
                        svg.triangle(x+cw-9, y+8, "#ffffff")
            for ix, gap in enumerate(gaps):
                svg.text(ox+ix*cw+cw/2, oy+len(eps)*ch+18, f"{gap:g}", 11, "middle")
            svg.text(ox+175, oy+len(eps)*ch+39, "Physical boundary gap Delta rho", 12, "middle")
            svg.text(ox-65, oy+105, "Error power 1 - R", 12, "middle", rotate=-90)
    legend(svg, 180, 774, 750, "angle")
    svg.triangle(1000, 782)
    svg.text(1015, 787, "Population slow subspace differs", 12)
    svg.text(32, 839, "R is signal power, not MSE-based state R2. Angles use oracle physical coordinates; eigenvector fitting uses no labels.", 12)
    svg.text(32, 860, "Independent residual rates: all .5 (top) or [.5,.5,.94,.94] (bottom). Finite-sample transitions need not coincide exactly with population boundaries.", 11, color="#586574")
    return svg.finish()


METHODS = [
    ("raw_linear_sfa", "Raw linear", "#626b73", None),
    ("poly2_observed_sfa", "Observed Poly2", "#007c91", None),
    ("oracle_latent_hermite_dictionary", "Oracle latent Hermite", "#8e44ad", "7 4"),
    ("identifiable_power_1", "Identifiable, R=1", "#d55e00", None),
]


def harmonic_figure(data):
    rows = [r for r in data["harmonic"]["rows"] if r["N_independent_lag1_pairs"] == 65536]
    lookup = {(r["case"], r["method"]): r for r in rows}
    rhos = [.87, .885, .895, .9025, .91, .925, .94]
    fields = ["function_subspace_sin_theta_max", "harmonic_H2_projection_R2"]
    labels = ["Function-space sin(theta_max)", "H2(Z0) projection R2"]
    svg = SVG(1460, 900, "Harmonic competition depends on the available feature dictionary",
              "N=65536 independent lag-one state pairs; independent AUDIT=16384 states. This is not 65536 trajectories.", data["seed"])
    for i, (_, name, color, dash) in enumerate(METHODS):
        x = 40+i*350
        svg.line(x, 108, x+35, 108, color, 3, dash)
        svg.text(x+45, 112, name, 13)
    for panel, field in enumerate(fields):
        ox, oy, width, height = 90+panel*715, 161, 595, 330
        svg.text(ox+width/2, oy-20, labels[panel], 17, "middle", weight="bold")
        for yval in [0, .2, .4, .6, .8, 1]:
            y = oy+height*(1-yval)
            svg.line(ox, y, ox+width, y, "#dfe6ec")
            svg.text(ox-10, y+4, f"{yval:.1f}", 11, "end")
        for rho in rhos:
            x = ox+(rho-rhos[0])/(rhos[-1]-rhos[0])*width
            svg.line(x, oy+height, x, oy+height+5, "#17212b")
            svg.text(x, oy+height+22, f"{rho:g}", 11, "middle")
        critical = ox+(.9025-rhos[0])/(rhos[-1]-rhos[0])*width
        svg.line(critical, oy, critical, oy+height, "#b52a2a", 1.5, "4 4")
        svg.text(critical+7, oy+14, "rho0^2 = .9025", 11, color="#b52a2a")
        for method, name, color, dash in METHODS:
            points = [(ox+(rho-rhos[0])/(rhos[-1]-rhos[0])*width,
                       oy+height*(1-lookup[f"rho1_{rho:g}", method][field])) for rho in rhos]
            svg.path(points, color, 2.5, dash)
            for x, y in points:
                svg.circle(x, y, 3.5, color, "white")
        svg.text(ox+width/2, oy+height+47, "rho1; rho0=.95, rho2=.86, rho3=.82", 12, "middle")
        # Reference cases have different rho0 and therefore separate axes.
        by, bh = 616, 145
        svg.text(ox+width/2, by-23, "Archived-rate references (rho0=.93)", 14, "middle", weight="bold")
        for yval in [0, .5, 1]:
            y = by+bh*(1-yval)
            svg.line(ox, y, ox+width, y, "#e5ebf0")
            svg.text(ox-10, y+4, f"{yval:g}", 10, "end")
        for ci, case in enumerate(["separated_wide", "separated_narrow"]):
            gx = ox+30+ci*310
            for mi, (method, _, color, _) in enumerate(METHODS):
                value = lookup[case, method][field]
                svg.rect(gx+mi*54, by+bh*(1-value), 40, bh*value, color, sw=0)
            svg.text(gx+101, by+bh+21, case.replace("separated_", ""), 12, "middle")
    svg.text(32, 834, "The vertical boundary is exact for the oracle latent Hermite span only. Observed Poly2 does not exactly contain Z0 or H2(Z0).", 12)
    svg.text(32, 856, "At equality, the population second eigenspace is not unique. Oracle dictionary and all recovery metrics use truth; fitting of other SFA arms is unlabeled.", 11)
    svg.text(32, 878, "One paired calibration draw per case/budget; no independent replication interval and no neural-network training claim.", 11, color="#586574")
    return svg.finish()


def stress_figure(data):
    stress = data["stress"]
    signals = stress["design"]["requested_fast_signal_fractions"]
    leaks = stress["design"]["requested_leak_amplitudes"]
    branches = stress["design"]["noise_branches"]
    lookup = {(r["branch"], r["fast_signal_fraction"], r["leakage_amplitude"]): r for r in stress["requested_cells"]}
    svg = SVG(1280, 700, "Fast-axis attenuation stress test: own-CAL selective error",
              "Requested grid only; slow-axis population state R2=.94; T=193; CAL=2048, TEST=4096, REF=32768.", stress["seed"])
    cw, ch = 110, 48
    for ib, (branch, sigma) in enumerate(branches.items()):
        ox, oy = 115+ib*610, 148
        svg.text(ox+220, oy-27, f"White-noise SD sigma={sigma:g}", 17, "middle", weight="bold")
        for iy, s in enumerate(signals):
            svg.text(ox-12, oy+iy*ch+29, f"{s:g}", 13, "end")
            for ix, leak in enumerate(leaks):
                row = lookup[branch, s, leak]
                value = row["own_cal"]["risk"]
                x, y = ox+ix*cw, oy+iy*ch
                svg.rect(x, y, cw, ch, risk_color(value), sw=1)
                color = "white" if risk_position(value)>.72 else "#17212b"
                svg.text(x+cw/2, y+29, f"{100*value:.2f}%", 14, "middle", color=color)
                if value < .01:
                    svg.circle(x+10, y+11, 2.8, "#202b36", "white")
                if not row["audit_gate_min_r2_ge_085"]:
                    svg.line(x+cw-16, y+7, x+cw-7, y+16, color, 1.7)
                    svg.line(x+cw-7, y+7, x+cw-16, y+16, color, 1.7)
        for ix, leak in enumerate(leaks):
            svg.text(ox+ix*cw+cw/2, oy+len(signals)*ch+23, f"{leak:g}", 12, "middle")
        svg.text(ox+220, oy+len(signals)*ch+48, "Slow leakage amplitude", 13, "middle")
        svg.text(ox-70, oy+170, "Fast signal fraction s", 13, "middle", rotate=-90)
    legend(svg, 115, 589, 660)
    svg.circle(830, 596, 3, "#202b36", "white")
    svg.text(842, 600, "Observed risk <1%", 12)
    svg.text(1034, 600, "x: AUDIT min R2 <.85", 12)
    svg.text(32, 659, "s is a signal fraction, NOT MSE-based state R2. All cells use their own predicted REF variance and own-CAL median-gap threshold.", 12)
    svg.text(32, 681, "Shared physical/noise paths support paired descriptions. Cell values are estimates, not simultaneous certificates; no trained H-JEPA evaluation.", 11, color="#586574")
    return svg.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate only this renderer's five SVGs and manifest")
    args = parser.parse_args()
    start = time.perf_counter()
    p1path, p2path = BASE/"probe1_summary.json", BASE/"probe2_summary.json"
    p1, p2 = json.loads(p1path.read_text()), json.loads(p2path.read_text())
    assert p1["evidence_tier"] == p2["evidence_tier"] == TIER
    figures = {"phase_all_risk.svg": phase_figure(p2, "all"),
               "phase_own_cal_risk.svg": phase_figure(p2, "own_cal"),
               "subspace_phase.svg": subspace_figure(p1),
               "harmonic_boundary.svg": harmonic_figure(p1),
               "stress.svg": stress_figure(p2)}
    folder = BASE/"figures"
    for name, source in figures.items():
        if (folder/name).exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing figure {folder/name}")
        root = ET.fromstring(source)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
    folder.mkdir(parents=True, exist_ok=True)
    for name, source in figures.items():
        (folder/name).write_text(source)
        ET.parse(folder/name)
    manifest = {"evidence_tier": TIER, "renderer": "Python standard library SVG; no plotting packages",
                "runtime_seconds": time.perf_counter()-start,
                "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (p1path, p2path)},
                "figures": {name: {"sha256": hashlib.sha256(source.encode()).hexdigest(), "xml_validated": True,
                                   "bytes": len(source.encode())} for name, source in figures.items()},
                "notes": ["All figures are descriptive paired numerical sandboxes.", "Below-1-percent marks use observed risk, not confidence bounds.",
                          "Harmonic oracle reference and subspace diagnostics use ground truth."]}
    (folder/"MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"evidence_tier": TIER, "figures": list(figures), "runtime_seconds": manifest["runtime_seconds"],
                      "XML_validation": "all five passed"}))


if __name__ == "__main__":
    main()
