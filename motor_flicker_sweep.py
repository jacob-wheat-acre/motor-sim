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
import math
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
def sag_percent(s: Scenario) -> float:
    # Thevenin impedance on HV side
    Z_line = complex(s.cond.r_ohm_per_mile * s.miles, s.cond.x_ohm_per_mile * s.miles)
    Z_xfmr = volt_sag.z_from_pctz(s.v_ll_hv, s.xfmr_kva, s.xfmr_pct_z, s.xfmr_x_over_r)
    Z_th = Z_line + Z_xfmr

    Vth_phase = complex(s.v_ll_hv / volt_sag.SQRT3, 0.0)

    # Capacitor bank input is 3-phase total kVAr at the load bus.
    Qcap_total_var = s.cap_kvar * 1000.0
    Qcap_phase_var = Qcap_total_var / 3.0

    if s.start_mode.upper() == "VFD":
        fla_lv = volt_sag.estimate_fla(s.hp, s.v_ll_lv, s.run_eff, s.run_pf)
        i_lim_lv = s.vfd_i_limit_pu_fla * fla_lv
        i_lim_hv = i_lim_lv * (s.v_ll_lv / s.v_ll_hv)

        pf = 0.95  # assumed
        I_motor = complex(i_lim_hv * pf, -i_lim_hv * math.sqrt(max(1.0 - pf * pf, 0.0)))

        # crude cap interaction
        Qm = abs(Vth_phase) * 3.0 * abs(I_motor) * math.sqrt(max(1.0 - pf * pf, 0.0))
        Qnet = max(Qm - Qcap_total_var, 0.0)
        Pm = abs(Vth_phase) * 3.0 * abs(I_motor) * pf
        Smag = math.sqrt(Pm * Pm + Qnet * Qnet)
        Imag = Smag / (3.0 * max(abs(Vth_phase), 1e-9))
        pf_new = Pm / max(Smag, 1e-9)
        I = complex(Imag * pf_new, -Imag * math.sqrt(max(1.0 - pf_new * pf_new, 0.0)))

        V_load = Vth_phase - Z_th * I

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
) -> np.ndarray:
    sag = np.zeros((len(miles_vals), len(hp_vals)), dtype=float)
    for i, miles in enumerate(miles_vals):
        for j, hp in enumerate(hp_vals):
            s = replace(base, miles=float(miles), hp=float(hp))
            sag[i, j] = sag_percent(s)
    return sag


def max_hp_under_threshold(
    base: Scenario,
    miles: float,
    hp_vals: np.ndarray,
    threshold_pct: float,
) -> float:
    sags = np.array([sag_percent(replace(base, miles=float(miles), hp=float(hp))) for hp in hp_vals])
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
) -> plt.Figure:
    limit_pct = allowable_dv_percent_mv(starts_per_hour)
    review_pct = review_fraction_of_limit * limit_pct
    sag = compute_sag_grid(base, hp_vals, miles_vals)

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
        f"Overview: HP vs Miles | {base.start_mode.upper()} | limit={limit_pct:.1f}% @ {starts_per_hour:.1f} starts/hr"
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
) -> plt.Figure:
    limit_pct = allowable_dv_percent_mv(starts_per_hour)
    review_pct = review_fraction_of_limit * limit_pct
    hp_scan_vals = hp_scan_vals if hp_scan_vals is not None else np.linspace(5, 600, 120)

    rows = []
    for label, cond in conductor_variants:
        s = replace(base, cond=cond, hp=float(selected_hp), miles=float(selected_miles))
        sag = sag_percent(s)
        max_hp_no_review = max_hp_under_threshold(replace(base, cond=cond), selected_miles, hp_scan_vals, review_pct)
        max_hp_within_limit = max_hp_under_threshold(replace(base, cond=cond), selected_miles, hp_scan_vals, limit_pct)

        if sag >= limit_pct:
            status = "LIKELY VIOLATION"
        elif sag >= review_pct:
            status = "REVIEW"
        else:
            status = "OK"

        rows.append(
            {
                "conductor": label,
                "sag": sag,
                "margin": limit_pct - sag,
                "status": status,
                "max_hp_no_review": max_hp_no_review,
                "max_hp_within_limit": max_hp_within_limit,
            }
        )

    rows.sort(key=lambda r: r["sag"])
    labels = [r["conductor"] for r in rows]
    sag_vals = [r["sag"] for r in rows]
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
    ax_bar.set_xlabel("Predicted start sag (%ΔV/V)")
    ax_bar.set_title(
        f"Compare Mode at {selected_hp:.0f} HP, {selected_miles:.1f} mi | {base.start_mode.upper()} | {starts_per_hour:.1f} starts/hr"
    )
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.legend(loc="lower right")

    table_data = [
        [
            r["conductor"],
            f'{r["sag"]:.2f}',
            f'{r["margin"]:+.2f}',
            r["status"],
            f'{r["max_hp_no_review"]:.0f}',
            f'{r["max_hp_within_limit"]:.0f}',
        ]
        for r in rows
    ]
    col_labels = ["Conductor", "Sag %", "Margin to Limit %", "Status", "Max HP (No Review)", "Max HP (Within Limit)"]
    ax_table.axis("off")
    tbl = ax_table.table(cellText=table_data, colLabels=col_labels, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Console output for quick screening use in engineering review workflow.
    print("\n=== Engineering screening heuristic ===")
    print(f"Selected point: {selected_hp:.1f} HP @ {selected_miles:.2f} mi, mode={base.start_mode.upper()}")
    print(f"Review threshold: {review_pct:.2f}% sag (80% of limit), hard limit: {limit_pct:.2f}% sag")
    print("If requested HP exceeds 'Max HP (No Review)', trigger detailed flicker review.")
    for r in rows:
        print(
            f'{r["conductor"]:>9}: sag={r["sag"]:5.2f}% | {r["status"]:<16} | '
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
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Prompt for selected motor HP, miles, and start technology at runtime.",
    )
    args = parser.parse_args()

    # ---- sweep ranges for overview and screening interpolation ----
    hp_vals = np.linspace(5, 400, 80)
    miles_vals = np.linspace(0.1, 12.0, 80)
    hp_scan_vals = np.linspace(5, 600, 120)

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

    # ---- common conductor set ----
    # Common ACSR values converted to ohms/mile from Encore ACSR table
    # using AC resistance at 25C and inductive reactance at 60 Hz, 1 ft equiv spacing.
    # (ohm/mile = ohm/kft * 5.28)
    conductor_variants = [
        ("795 ACSR", Conductor(0.119, 0.412)),
        ("336 ACSR", Conductor(0.276, 0.463)),
        ("2/0 ACSR", Conductor(0.686, 0.537)),
        ("#2 ACSR", Conductor(1.367, 0.581)),
        ("#4 ACSR", Conductor(2.175, 0.608)),
    ]

    selected_hp = args.motor_hp
    selected_miles = args.miles
    selected_start_tech = args.start_tech
    starts_per_hour = args.starts_per_hour
    selected_cap_kvar = args.cap_kvar

    if args.prompt:
        selected_hp = prompt_or_default_float("Motor size (HP)", selected_hp)
        selected_miles = prompt_or_default_float("Feeder miles", selected_miles)
        selected_start_tech = prompt_or_default_text("Starting technology (ATL, SoftStart, VFD)", selected_start_tech)
        has_cap_bank = prompt_yes_no("Is there a nearby capacitor bank installed?", default_yes=(selected_cap_kvar > 0))
        if has_cap_bank:
            default_kvar = selected_cap_kvar if selected_cap_kvar > 0 else 300.0
            selected_cap_kvar = prompt_or_default_float("Capacitor bank size (kVAr, 3-phase total)", default_kvar)
        else:
            selected_cap_kvar = 0.0

    selected_start_tech = parse_start_tech(selected_start_tech)
    review_fraction_of_limit = 0.8

    # ---- mode 1: one map + risk overlay ----
    # Use #2 ACSR as baseline conductor for map view; apply selected start technology.
    overview_base = apply_start_tech(
        replace(base, cond=conductor_variants[3][1], cap_kvar=selected_cap_kvar),
        selected_start_tech,
    )
    overview_mode(
        base=overview_base,
        hp_vals=hp_vals,
        miles_vals=miles_vals,
        starts_per_hour=starts_per_hour,
        review_fraction_of_limit=review_fraction_of_limit,
        vmax=6.0,
    )

    # ---- mode 2: one point + conductor comparison + ranked table ----
    compare_mode(
        base=overview_base,
        conductor_variants=conductor_variants,
        selected_hp=selected_hp,
        selected_miles=selected_miles,
        starts_per_hour=starts_per_hour,
        review_fraction_of_limit=review_fraction_of_limit,
        hp_scan_vals=hp_scan_vals,
    )

    plt.show()


if __name__ == "__main__":
    main()
