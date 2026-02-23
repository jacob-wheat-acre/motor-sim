#!/usr/bin/env python3
"""
motor_flicker_sweep.py

Sweep motor HP vs feeder miles and plot:
  - predicted sag (%ΔV/V) during start (from volt_sag Scenario model)
  - "risk" relative to Xcel MV rapid voltage change limit (step-change table)

Provides two plotting modes:
  - overview_mode(): one HP-vs-miles heatmap + risk overlay
  - compare_mode(): one selected point + conductor bar chart + ranked table
"""

from __future__ import annotations

import argparse
import cmath
import math
import platform
import sys
from dataclasses import replace
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import volt_sag
from volt_sag import Scenario, Conductor

START_TECH_ALIASES = {
    "atl": "ATL",
    "across_the_line": "ATL",
    "softstart": "SOFTSTART",
    "soft_start": "SOFTSTART",
    "vfd": "VFD",
}

# Common ACSR values converted to ohms/mile from Encore ACSR table
# using AC resistance at 25C and inductive reactance at 60 Hz, 1 ft equiv spacing.
# (ohm/mile = ohm/kft * 5.28)
CONDUCTOR_VARIANTS: List[Tuple[str, Conductor]] = [
    ("795 ACSR", Conductor(0.119, 0.412)),
    ("336 ACSR", Conductor(0.276, 0.463)),
    ("2/0 ACSR", Conductor(0.686, 0.537)),
    ("#2 ACSR", Conductor(1.367, 0.581)),
    ("#4 ACSR", Conductor(2.175, 0.608)),
]


# -----------------------------
# Xcel MV rapid voltage change limits (step-changes)
# MV column summary (conservative choices):
#   N <= 4 per day: 5% (table shows 5–6%; use 5 for conservative)
#   N <= 2 per hour: 4%
#   2 < N <= 10 per hour: 3%
# -----------------------------
def allowable_dv_percent_mv(
    starts_per_hour: float,
    starts_per_day: Optional[float] = None,
    conservative_daily_pct: float = 5.0,
) -> float:
    if starts_per_day is None:
        starts_per_day = starts_per_hour * 24.0

    # Infrequent events: allow daily row
    if starts_per_day <= 4.0:
        return conservative_daily_pct

    # Otherwise use hourly rows
    if starts_per_hour <= 2.0:
        return 4.0
    if starts_per_hour <= 10.0:
        return 3.0

    # Beyond 10/hr: already deep flicker territory; hold conservative at 3%
    return 3.0


# -----------------------------
# Compute sag % using the same math as volt_sag.run(), but return a number.
# -----------------------------
def sag_percent(
    s: Scenario,
    z_line_override: Optional[complex] = None,
    z_upstream_phase: complex = 0j,
) -> float:
    # Thevenin impedance on HV side
    Z_line = z_line_override if z_line_override is not None else complex(s.cond.r_ohm_per_mile * s.miles, s.cond.x_ohm_per_mile * s.miles)
    Z_xfmr = volt_sag.z_from_pctz(s.v_ll_hv, s.xfmr_kva, s.xfmr_pct_z, s.xfmr_x_over_r)
    Z_th = z_upstream_phase + Z_line + Z_xfmr

    Vth_phase = complex(s.v_ll_hv / volt_sag.SQRT3, 0.0)

    # Capacitor bank input is 3-phase total kVAr at the load bus.
    Qcap_total_var = s.cap_kvar * 1000.0
    Qcap_phase_var = Qcap_total_var / 3.0

    if s.start_mode.upper() == "VFD":
        fla_lv = volt_sag.estimate_fla(s.hp, s.v_ll_lv, s.run_eff, s.run_pf)
        i_lim_lv = s.vfd_i_limit_pu_fla * fla_lv
        i_lim_hv = i_lim_lv * (s.v_ll_lv / s.v_ll_hv)

        # Solve VFD input current against local bus voltage phasor:
        # - motor current magnitude limited by VFD current limit
        # - motor displacement PF assumed near-unity but lagging
        # - capacitor current computed from local phasor voltage
        pf = 0.95
        phi = math.acos(pf)
        V = Vth_phase
        for _ in range(80):
            theta_v = cmath.phase(V)
            I_motor = cmath.rect(i_lim_hv, theta_v - phi)
            v_conj = V.conjugate() if abs(V) > 1e-9 else complex(1e-9, 0.0)
            Icap = complex(0.0, +Qcap_phase_var) / v_conj
            I_total = I_motor + Icap
            V_new = Vth_phase - Z_th * I_total
            if abs(V_new - V) / max(abs(V), 1e-9) < 1e-7:
                V = V_new
                break
            V = V_new
        V_load = V

    else:
        # ATL
        S_lr_kva = volt_sag.locked_rotor_kva(s.hp, s.motor_code)

        if s.atl_model.upper() == "CONST_S":
            S = S_lr_kva * 1000.0
            P = S * s.start_pf
            Q = math.sqrt(max(S * S - P * P, 0.0))
            Q_net = Q - Qcap_total_var
            S3 = complex(P, Q_net)
            V_load = volt_sag.two_bus_solve_constS(Vth_phase, Z_th, S3)

        elif s.atl_model.upper() == "CONST_I":
            I_lra_hv = (S_lr_kva * 1000.0) / (volt_sag.SQRT3 * s.v_ll_hv)
            I_mag = s.atl_I_multiplier_of_LRA * I_lra_hv

            pf = s.start_pf
            I_motor = complex(I_mag * pf, -I_mag * math.sqrt(max(1.0 - pf * pf, 0.0)))

            V = Vth_phase
            for _ in range(60):
                Icap = complex(0.0, +Qcap_phase_var / max(abs(V), 1e-9))
                I_total = I_motor + Icap
                V_new = Vth_phase - Z_th * I_total
                if abs(V_new - V) / max(abs(V), 1e-9) < 1e-7:
                    V = V_new
                    break
                V = V_new
            V_load = V

        else:
            raise ValueError("atl_model must be CONST_S or CONST_I")

    return 100.0 * (1.0 - abs(V_load) / abs(Vth_phase))


def compute_sag_grid(
    base: Scenario,
    hp_vals: np.ndarray,
    miles_vals: np.ndarray,
    z_upstream_phase: complex = 0j,
) -> np.ndarray:
    sag = np.zeros((len(miles_vals), len(hp_vals)), dtype=float)
    for i, miles in enumerate(miles_vals):
        for j, hp in enumerate(hp_vals):
            s = replace(base, miles=float(miles), hp=float(hp))
            # Flicker screening is based on magnitude of rapid voltage change (rise or sag).
            sag[i, j] = abs(sag_percent(s, z_upstream_phase=z_upstream_phase))
    return sag


def max_hp_under_threshold(
    base: Scenario,
    miles: float,
    hp_vals: np.ndarray,
    threshold_pct: float,
    z_line_override: Optional[complex] = None,
    z_upstream_phase: complex = 0j,
) -> float:
    sags = np.array(
        [
            abs(
                sag_percent(
                    replace(base, miles=float(miles), hp=float(hp)),
                    z_line_override=z_line_override,
                    z_upstream_phase=z_upstream_phase,
                )
            )
            for hp in hp_vals
        ]
    )
    over = np.where(sags > threshold_pct)[0]
    if len(over) == 0:
        return float(hp_vals[-1])
    k = int(over[0])
    if k == 0:
        return 0.0
    h1, h2 = float(hp_vals[k - 1]), float(hp_vals[k])
    y1, y2 = float(sags[k - 1]), float(sags[k])
    if abs(y2 - y1) < 1e-9:
        return h1
    frac = (threshold_pct - y1) / (y2 - y1)
    return h1 + frac * (h2 - h1)


def overview_mode(
    base: Scenario,
    hp_vals: np.ndarray,
    miles_vals: np.ndarray,
    starts_per_hour: float = 4.0,
    review_fraction_of_limit: float = 0.8,
    vmax: float = 6.0,
    z_upstream_phase: complex = 0j,
) -> plt.Figure:
    limit_pct = allowable_dv_percent_mv(starts_per_hour)
    review_pct = review_fraction_of_limit * limit_pct
    sag = compute_sag_grid(base, hp_vals, miles_vals, z_upstream_phase=z_upstream_phase)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(
        sag,
        origin="lower",
        aspect="auto",
        extent=[hp_vals.min(), hp_vals.max(), miles_vals.min(), miles_vals.max()],
        vmin=0.0,
        vmax=vmax,
        cmap="viridis",
    )
    cbar = fig.colorbar(im, ax=ax, label="Start sag (%ΔV/V)")

    # Risk classes overlay: green (ok), yellow (review), red (likely violation)
    overlay_levels = [0.0, review_pct, limit_pct, max(vmax, float(np.max(sag)) + 0.1)]
    ax.contourf(
        hp_vals,
        miles_vals,
        sag,
        levels=overlay_levels,
        colors=["#2ca02c", "#ffbf00", "#d62728"],
        alpha=0.18,
    )
    c_review = ax.contour(hp_vals, miles_vals, sag, levels=[review_pct], colors=["#b8860b"], linestyles=["--"], linewidths=1.8)
    c_limit = ax.contour(hp_vals, miles_vals, sag, levels=[limit_pct], colors=["#8b0000"], linestyles=["-"], linewidths=2.2)
    cbar.ax.text(1.6, 0.96, f"Review: {review_pct:.1f}%", transform=cbar.ax.transAxes, fontsize=8, va="top")
    cbar.ax.text(1.6, 0.90, f"Limit: {limit_pct:.1f}%", transform=cbar.ax.transAxes, fontsize=8, va="top")
    ax.clabel(c_review, fmt={review_pct: f"review {review_pct:.1f}%"}, inline=True, fontsize=8)
    ax.clabel(c_limit, fmt={limit_pct: f"limit {limit_pct:.1f}%"}, inline=True, fontsize=8)

    ax.set_title(
        f"Overview: HP vs Miles | {base.start_mode.upper()} | xfmr={base.xfmr_kva:.0f} kVA | "
        f"cap={base.cap_kvar:.0f} kVAr | limit={limit_pct:.1f}% @ {starts_per_hour:.1f} starts/hr"
    )
    ax.set_xlabel("Motor size (HP)")
    ax.set_ylabel("Feeder miles (approx)")
    ax.text(
        0.01,
        0.01,
        "Heuristic: Yellow/Red regions should trigger engineering review.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    return fig


def compare_mode(
    base: Scenario,
    conductor_variants: List[Tuple[str, Conductor]],
    selected_hp: float,
    selected_miles: float,
    starts_per_hour: float = 4.0,
    review_fraction_of_limit: float = 0.8,
    hp_scan_vals: Optional[np.ndarray] = None,
    z_upstream_phase: complex = 0j,
) -> plt.Figure:
    limit_pct = allowable_dv_percent_mv(starts_per_hour)
    review_pct = review_fraction_of_limit * limit_pct
    hp_scan_vals = hp_scan_vals if hp_scan_vals is not None else np.linspace(5, 600, 120)

    rows = []
    for label, cond in conductor_variants:
        s = replace(base, cond=cond, hp=float(selected_hp), miles=float(selected_miles))
        sag_signed = sag_percent(s, z_upstream_phase=z_upstream_phase)
        dv_abs = abs(sag_signed)
        max_hp_no_review = max_hp_under_threshold(
            replace(base, cond=cond),
            selected_miles,
            hp_scan_vals,
            review_pct,
            z_upstream_phase=z_upstream_phase,
        )
        max_hp_within_limit = max_hp_under_threshold(
            replace(base, cond=cond),
            selected_miles,
            hp_scan_vals,
            limit_pct,
            z_upstream_phase=z_upstream_phase,
        )

        if dv_abs >= limit_pct:
            status = "LIKELY VIOLATION"
        elif dv_abs >= review_pct:
            status = "REVIEW"
        else:
            status = "OK"

        rows.append(
            {
                "conductor": label,
                "sag_signed": sag_signed,
                "dv_abs": dv_abs,
                "margin": limit_pct - dv_abs,
                "status": status,
                "max_hp_no_review": max_hp_no_review,
                "max_hp_within_limit": max_hp_within_limit,
            }
        )

    rows.sort(key=lambda r: r["dv_abs"])
    labels = [r["conductor"] for r in rows]
    sag_vals = [r["dv_abs"] for r in rows]
    colors = [
        "#d62728" if r["status"] == "LIKELY VIOLATION" else "#ffbf00" if r["status"] == "REVIEW" else "#2ca02c"
        for r in rows
    ]

    fig, (ax_bar, ax_table) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )
    ax_bar.barh(labels, sag_vals, color=colors)
    ax_bar.axvline(review_pct, color="#b8860b", linestyle="--", linewidth=1.8, label=f"Review {review_pct:.1f}%")
    ax_bar.axvline(limit_pct, color="#8b0000", linestyle="-", linewidth=2.0, label=f"Limit {limit_pct:.1f}%")
    ax_bar.set_xlabel("Predicted start |ΔV| (%V/V)")
    ax_bar.set_title(
        f"Compare at {selected_hp:.0f} HP, {selected_miles:.1f} mi | {base.start_mode.upper()} | "
        f"xfmr={base.xfmr_kva:.0f} kVA | cap={base.cap_kvar:.0f} kVAr | {starts_per_hour:.1f} starts/hr"
    )
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.legend(loc="lower right")

    table_data = [
        [
            r["conductor"],
            f'{r["sag_signed"]:+.2f}',
            f'{r["dv_abs"]:.2f}',
            f'{r["margin"]:+.2f}',
            r["status"],
            f'{r["max_hp_no_review"]:.0f}',
            f'{r["max_hp_within_limit"]:.0f}',
        ]
        for r in rows
    ]
    col_labels = [
        "Conductor",
        "Signed ΔV %",
        "|ΔV| %",
        "Margin to Limit %",
        "Status",
        "Max HP (No Review)",
        "Max HP (Within Limit)",
    ]
    ax_table.axis("off")
    tbl = ax_table.table(cellText=table_data, colLabels=col_labels, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Console output for quick screening use in engineering review workflow.
    print("\n=== Engineering screening heuristic ===")
    print(f"Selected point: {selected_hp:.1f} HP @ {selected_miles:.2f} mi, mode={base.start_mode.upper()}")
    if selected_hp > float(hp_scan_vals[-1]):
        print(f"Note: selected HP is above scan range ({hp_scan_vals[-1]:.0f} HP); Max HP columns are capped by scan range.")
    print(f"Review threshold: {review_pct:.2f}% |ΔV| (80% of limit), hard limit: {limit_pct:.2f}% |ΔV|")
    print("Screening uses |ΔV| (magnitude of voltage change), not signed sag only.")
    print("If requested HP exceeds 'Max HP (No Review)', trigger detailed flicker review.")
    for r in rows:
        print(
            f'{r["conductor"]:>9}: signed={r["sag_signed"]:+5.2f}% | |ΔV|={r["dv_abs"]:5.2f}% | {r["status"]:<16} | '
            f'NoReview<= {r["max_hp_no_review"]:6.1f} HP | WithinLimit<= {r["max_hp_within_limit"]:6.1f} HP'
        )

    return fig


def parse_start_tech(raw: str) -> str:
    key = raw.strip().lower()
    if key not in START_TECH_ALIASES:
        raise ValueError("start tech must be one of: ATL, SoftStart, VFD")
    return START_TECH_ALIASES[key]


def apply_start_tech(base: Scenario, start_tech: str) -> Scenario:
    if start_tech == "ATL":
        return replace(base, start_mode="ATL", atl_I_multiplier_of_LRA=1.0)
    if start_tech == "SOFTSTART":
        return replace(base, start_mode="ATL", atl_I_multiplier_of_LRA=0.5)
    if start_tech == "VFD":
        return replace(base, start_mode="VFD", atl_I_multiplier_of_LRA=1.0)
    raise ValueError("unsupported start technology")


def prompt_or_default_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return float(raw)


def prompt_or_default_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def prompt_yes_no(prompt: str, default_yes: bool = False) -> bool:
    default_text = "Y/n" if default_yes else "y/N"
    raw = input(f"{prompt} ({default_text}): ").strip().lower()
    if not raw:
        return default_yes
    return raw in ("y", "yes")


def print_environment_info() -> None:
    print("=== Environment ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Executable: {sys.executable}")
    print(f"NumPy: {np.__version__}")
    print(f"Matplotlib: {plt.matplotlib.__version__}")


def prompt_conductor_miles(conductor_variants: List[Tuple[str, Conductor]]) -> List[Tuple[str, Conductor, float]]:
    print("\nEnter segment miles by conductor type (press Enter for 0.0):")
    segments: List[Tuple[str, Conductor, float]] = []
    for label, cond in conductor_variants:
        miles = prompt_or_default_float(f"  {label} miles", 0.0)
        if miles > 0.0:
            segments.append((label, cond, miles))
    return segments


def line_impedance_from_segments(segments: List[Tuple[str, Conductor, float]]) -> complex:
    z = 0j
    for _, cond, miles in segments:
        z += complex(cond.r_ohm_per_mile * miles, cond.x_ohm_per_mile * miles)
    return z


def status_from_dv(dv_abs: float, review_pct: float, limit_pct: float) -> str:
    if dv_abs >= limit_pct:
        return "LIKELY VIOLATION"
    if dv_abs >= review_pct:
        return "REVIEW"
    return "OK"


def print_ascii_study(
    s: Scenario,
    starts_per_hour: float,
    z_upstream: complex,
    segments: List[Tuple[str, Conductor, float]],
    z_line_total: complex,
    sag_signed: float,
    dv_abs: float,
    status: str,
) -> None:
    z_xfmr = volt_sag.z_from_pctz(s.v_ll_hv, s.xfmr_kva, s.xfmr_pct_z, s.xfmr_x_over_r)
    z_total = z_upstream + z_xfmr + z_line_total
    limit_pct = allowable_dv_percent_mv(starts_per_hour)
    review_pct = 0.8 * limit_pct

    print("\n=== Custom Study One-Line (ASCII) ===")
    print("[Utility Source] -- Z_upstream -- Z_xfmr -- Z_line_segments -- [Motor/VFD + Cap Bank]")
    print(f"  Z_upstream (ohm/ph): {z_upstream.real:.4f} + j{z_upstream.imag:.4f}")
    print(f"  Z_xfmr     (ohm/ph): {z_xfmr.real:.4f} + j{z_xfmr.imag:.4f}")
    print(f"  Z_line     (ohm/ph): {z_line_total.real:.4f} + j{z_line_total.imag:.4f}")
    print(f"  Z_th total (ohm/ph): {z_total.real:.4f} + j{z_total.imag:.4f}")
    print("  Line segments:")
    if segments:
        for label, cond, miles in segments:
            zr = cond.r_ohm_per_mile * miles
            zx = cond.x_ohm_per_mile * miles
            print(f"    - {label:8s}: {miles:6.3f} mi -> {zr:7.4f} + j{zx:7.4f} ohm")
    else:
        print("    - (none entered, line impedance = 0)")
    print(
        f"  Study point: {s.hp:.1f} HP, mode={s.start_mode.upper()}, cap={s.cap_kvar:.1f} kVAr (3-ph), "
        f"xfmr={s.xfmr_kva:.0f} kVA, starts/hr={starts_per_hour:.2f}"
    )
    print(f"  Limits: review={review_pct:.2f}% |ΔV|, hard={limit_pct:.2f}% |ΔV|")
    print(f"  Result: signed ΔV={sag_signed:+.2f}%  |ΔV|={dv_abs:.2f}%  status={status}")


def custom_study_mode(
    base: Scenario,
    starts_per_hour: float,
    hp_scan_vals: np.ndarray,
    z_upstream_phase: complex,
    segments: List[Tuple[str, Conductor, float]],
) -> plt.Figure:
    z_line_total = line_impedance_from_segments(segments)
    s = replace(base, miles=0.0)

    limit_pct = allowable_dv_percent_mv(starts_per_hour)
    review_pct = 0.8 * limit_pct
    sag_signed = sag_percent(s, z_line_override=z_line_total, z_upstream_phase=z_upstream_phase)
    dv_abs = abs(sag_signed)
    status = status_from_dv(dv_abs, review_pct, limit_pct)
    color = "#d62728" if status == "LIKELY VIOLATION" else "#ffbf00" if status == "REVIEW" else "#2ca02c"

    max_hp_no_review = max_hp_under_threshold(
        s,
        0.0,
        hp_scan_vals,
        review_pct,
        z_line_override=z_line_total,
        z_upstream_phase=z_upstream_phase,
    )
    max_hp_within_limit = max_hp_under_threshold(
        s,
        0.0,
        hp_scan_vals,
        limit_pct,
        z_line_override=z_line_total,
        z_upstream_phase=z_upstream_phase,
    )

    fig, (ax_bar, ax_tbl) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1.6, 1.4]})
    ax_bar.barh(["Study case"], [dv_abs], color=[color], height=0.45)
    ax_bar.axvline(review_pct, color="#b8860b", linestyle="--", linewidth=1.8, label=f"Review {review_pct:.1f}%")
    ax_bar.axvline(limit_pct, color="#8b0000", linestyle="-", linewidth=2.0, label=f"Limit {limit_pct:.1f}%")
    ax_bar.set_xlim(0, max(limit_pct * 1.8, dv_abs * 1.15, 1.0))
    ax_bar.set_xlabel("Rapid voltage change |ΔV| (%)")
    ax_bar.set_title(
        f"Custom Study | {s.hp:.0f} HP {s.start_mode.upper()} | xfmr={s.xfmr_kva:.0f} kVA | "
        f"cap={s.cap_kvar:.0f} kVAr | starts/hr={starts_per_hour:.1f}"
    )
    ax_bar.legend(loc="lower right")
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.text(
        dv_abs,
        0,
        f"  signed ΔV={sag_signed:+.2f}%\n  status={status}",
        va="center",
        fontsize=9,
    )

    z_xfmr = volt_sag.z_from_pctz(s.v_ll_hv, s.xfmr_kva, s.xfmr_pct_z, s.xfmr_x_over_r)
    z_total = z_upstream_phase + z_xfmr + z_line_total
    segment_text = ", ".join([f"{label}:{miles:.2f}mi" for label, _, miles in segments]) or "(none)"
    rows = [
        ["Upstream Z (ohm/ph)", f"{z_upstream_phase.real:.4f} + j{z_upstream_phase.imag:.4f}"],
        ["Transformer Z (ohm/ph)", f"{z_xfmr.real:.4f} + j{z_xfmr.imag:.4f}"],
        ["Line Z total (ohm/ph)", f"{z_line_total.real:.4f} + j{z_line_total.imag:.4f}"],
        ["Thevenin Z total (ohm/ph)", f"{z_total.real:.4f} + j{z_total.imag:.4f}"],
        ["Segments", segment_text],
        ["Max HP (No Review)", f"{max_hp_no_review:.0f}"],
        ["Max HP (Within Limit)", f"{max_hp_within_limit:.0f}"],
    ]
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=rows, colLabels=["Study Input / Output", "Value"], loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)

    print_ascii_study(
        s=s,
        starts_per_hour=starts_per_hour,
        z_upstream=z_upstream_phase,
        segments=segments,
        z_line_total=z_line_total,
        sag_signed=sag_signed,
        dv_abs=dv_abs,
        status=status,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Motor flicker screening with overview and compare modes."
    )
    parser.add_argument("--motor-hp", type=float, default=200.0, help="Selected motor size for compare mode.")
    parser.add_argument("--miles", type=float, default=7.0, help="Selected feeder miles for compare mode.")
    parser.add_argument(
        "--start-tech",
        type=str,
        default="ATL",
        help="Starting technology: ATL, SoftStart, or VFD.",
    )
    parser.add_argument("--starts-per-hour", type=float, default=4.0, help="Expected starts per hour.")
    parser.add_argument(
        "--cap-kvar",
        type=float,
        default=0.0,
        help="Nearby 3-phase capacitor bank size in kVAr (0 if none).",
    )
    parser.add_argument("--xfmr-kva", type=float, default=500.0, help="Transformer kVA.")
    parser.add_argument("--xfmr-pct-z", type=float, default=5.75, help="Transformer percent impedance.")
    parser.add_argument("--xfmr-xr", type=float, default=10.0, help="Transformer X/R ratio.")
    parser.add_argument(
        "--upstream-r-ohm",
        type=float,
        default=0.0,
        help="Upstream Thevenin resistance (ohm/phase on MV side). Default 0.0.",
    )
    parser.add_argument(
        "--upstream-x-ohm",
        type=float,
        default=0.0,
        help="Upstream Thevenin reactance (ohm/phase on MV side). Default 0.0.",
    )
    parser.add_argument(
        "--hp-max",
        type=float,
        default=10000.0,
        help="Maximum HP used for overview/threshold scan ranges.",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Prompt for selected motor HP, miles, and start technology at runtime.",
    )
    parser.add_argument(
        "--case-study",
        action="store_true",
        help="Run one custom feeder study (segment miles + upstream Z) instead of comparison mode.",
    )
    parser.add_argument(
        "--study",
        action="store_true",
        help="Alias for --case-study.",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print Python/OS/package versions and exit.",
    )
    args = parser.parse_args()

    if args.print_env:
        print_environment_info()
        return

    # ---- sweep ranges for overview and screening interpolation ----
    hp_max = max(args.hp_max, 50.0)
    hp_vals = np.linspace(5, hp_max, 100)
    miles_vals = np.linspace(0.1, 12.0, 80)
    hp_scan_vals = np.linspace(5, hp_max, 220)

    # ---- scenario defaults ----
    base = Scenario(
        hp=50.0,
        motor_code="K",                  # try G/H/J/K
        miles=7.0,
        cond=Conductor(1.367, 0.581),    # #2 ACSR approx
        xfmr_kva=500.0,
        xfmr_pct_z=5.75,
        xfmr_x_over_r=10.0,
        start_mode="ATL",                # "ATL" or "VFD"
        atl_model="CONST_I",
        start_pf=0.25,
        atl_I_multiplier_of_LRA=1.0,
        vfd_i_limit_pu_fla=1.30,
        cap_kvar=0.0,
    )

    selected_hp = args.motor_hp
    selected_miles = args.miles
    selected_start_tech = args.start_tech
    starts_per_hour = args.starts_per_hour
    selected_cap_kvar = args.cap_kvar
    selected_xfmr_kva = args.xfmr_kva
    selected_xfmr_pct_z = args.xfmr_pct_z
    selected_xfmr_xr = args.xfmr_xr
    upstream_r_ohm = args.upstream_r_ohm
    upstream_x_ohm = args.upstream_x_ohm
    use_case_study = args.case_study or args.study

    if args.prompt or use_case_study:
        selected_hp = prompt_or_default_float("Motor size (HP)", selected_hp)
        if not use_case_study:
            selected_miles = prompt_or_default_float("Feeder miles", selected_miles)
        selected_start_tech = prompt_or_default_text("Starting technology (ATL, SoftStart, VFD)", selected_start_tech)
        has_cap_bank = prompt_yes_no("Is there a nearby capacitor bank installed?", default_yes=(selected_cap_kvar > 0))
        if has_cap_bank:
            default_kvar = selected_cap_kvar if selected_cap_kvar > 0 else 300.0
            selected_cap_kvar = prompt_or_default_float("Capacitor bank size (kVAr, 3-phase total)", default_kvar)
        else:
            selected_cap_kvar = 0.0
        selected_xfmr_kva = prompt_or_default_float("Transformer size (kVA)", selected_xfmr_kva)
        selected_xfmr_pct_z = prompt_or_default_float("Transformer percent impedance (%Z)", selected_xfmr_pct_z)
        selected_xfmr_xr = prompt_or_default_float("Transformer X/R", selected_xfmr_xr)
        upstream_r_ohm = prompt_or_default_float("Upstream Thevenin R (ohm/phase at MV)", upstream_r_ohm)
        upstream_x_ohm = prompt_or_default_float("Upstream Thevenin X (ohm/phase at MV)", upstream_x_ohm)

    selected_start_tech = parse_start_tech(selected_start_tech)
    z_upstream_phase = complex(upstream_r_ohm, upstream_x_ohm)
    review_fraction_of_limit = 0.8

    # ---- mode 1: one map + risk overlay ----
    # Use #2 ACSR as baseline conductor for map view; apply selected start technology.
    overview_base = apply_start_tech(
        replace(
            base,
            cond=CONDUCTOR_VARIANTS[3][1],
            cap_kvar=selected_cap_kvar,
            xfmr_kva=selected_xfmr_kva,
            xfmr_pct_z=selected_xfmr_pct_z,
            xfmr_x_over_r=selected_xfmr_xr,
        ),
        selected_start_tech,
    )

    if use_case_study:
        segments = prompt_conductor_miles(CONDUCTOR_VARIANTS)
        custom_study_mode(
            base=replace(overview_base, hp=selected_hp),
            starts_per_hour=starts_per_hour,
            hp_scan_vals=hp_scan_vals,
            z_upstream_phase=z_upstream_phase,
            segments=segments,
        )
    else:
        overview_mode(
            base=overview_base,
            hp_vals=hp_vals,
            miles_vals=miles_vals,
            starts_per_hour=starts_per_hour,
            review_fraction_of_limit=review_fraction_of_limit,
            vmax=6.0,
            z_upstream_phase=z_upstream_phase,
        )

        # ---- mode 2: one point + conductor comparison + ranked table ----
        compare_mode(
            base=overview_base,
            conductor_variants=CONDUCTOR_VARIANTS,
            selected_hp=selected_hp,
            selected_miles=selected_miles,
            starts_per_hour=starts_per_hour,
            review_fraction_of_limit=review_fraction_of_limit,
            hp_scan_vals=hp_scan_vals,
            z_upstream_phase=z_upstream_phase,
        )

    plt.show()


if __name__ == "__main__":
    main()
