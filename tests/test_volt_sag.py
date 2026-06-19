"""
Regression and property tests for volt_sag.py.

Reference values for spot checks were computed analytically or from a
known-good run and are locked here to catch solver regressions.
"""

import math
import pytest
from dataclasses import replace

from volt_sag import (
    NEMA_KVA_PER_HP,
    Conductor,
    Scenario,
    estimate_fla,
    locked_rotor_kva,
    run_return,
    solve_load_voltage,
    z_from_pctz,
    SQRT3,
)

# ---------------------------------------------------------------------------
# Shared fixture: simple 100 HP ATL case with no line, no cap, no upstream Z.
# Used as the baseline for property tests.
# ---------------------------------------------------------------------------
BASE = Scenario(
    hp=100.0,
    motor_code="G",
    v_ll_hv=12_470.0,
    v_ll_lv=480.0,
    miles=0.0,
    cond=Conductor(1.367, 0.581),
    xfmr_kva=500.0,
    xfmr_pct_z=5.75,
    xfmr_x_over_r=10.0,
    start_mode="ATL",
    atl_model="CONST_I",
    start_pf=0.25,
    atl_I_multiplier_of_LRA=1.0,
    cap_kvar=0.0,
)

# Analytically verified: 6.751 % for the BASE scenario above.
SAG_BASE_PCT = 6.751


# ---------------------------------------------------------------------------
# NEMA code table
# ---------------------------------------------------------------------------

class TestNemaTable:
    def test_full_range_present(self):
        for letter in "ABCDEFGHJKLMNPRSTU":
            assert letter in NEMA_KVA_PER_HP, f"Missing NEMA code {letter}"
        assert "V" in NEMA_KVA_PER_HP

    def test_bands_monotone(self):
        # Each band's upper bound should be >= its lower bound
        for letter, (lo, hi) in NEMA_KVA_PER_HP.items():
            assert hi >= lo, f"Band {letter}: hi < lo"

    def test_locked_rotor_kva_midpoint_g(self):
        # G: (5.60, 6.29) → midpoint = 5.945 kVA/HP → 100 HP = 594.5 kVA
        assert abs(locked_rotor_kva(100.0, "G") - 594.5) < 0.01

    def test_locked_rotor_kva_midpoint_k(self):
        # K: (8.00, 8.99) → midpoint = 8.495 kVA/HP → 100 HP = 849.5 kVA
        assert abs(locked_rotor_kva(100.0, "K") - 849.5) < 0.01

    def test_locked_rotor_kva_midpoint_e(self):
        # E: (4.50, 4.99) → midpoint = 4.745 kVA/HP → 200 HP = 949.0 kVA
        assert abs(locked_rotor_kva(200.0, "E") - 949.0) < 0.01

    def test_case_insensitive(self):
        assert locked_rotor_kva(100.0, "g") == locked_rotor_kva(100.0, "G")


# ---------------------------------------------------------------------------
# z_from_pctz
# ---------------------------------------------------------------------------

class TestZFromPctZ:
    def test_known_value(self):
        # 500 kVA, 12.47 kV, 5.75 %, X/R=10
        # z_base = 12470² / (500·1000) = 311.002 Ω
        # z_mag  = 0.0575 · 311.002  = 17.883 Ω
        # R = 17.883 / sqrt(101) = 1.779 Ω   X = 17.79 Ω
        z = z_from_pctz(12_470.0, 500.0, 5.75, 10.0)
        assert abs(z.real - 1.779) < 0.002
        assert abs(z.imag - 17.79) < 0.02

    def test_zero_resistance_at_infinite_xr(self):
        # Very high X/R → R → 0, X → z_mag
        z = z_from_pctz(12_470.0, 500.0, 5.75, 1e6)
        assert z.real < 1e-3
        assert z.imag > 0

    def test_scales_with_kva(self):
        # Larger transformer → smaller impedance in ohms (same %Z)
        z_small = z_from_pctz(12_470.0, 500.0, 5.75, 10.0)
        z_large = z_from_pctz(12_470.0, 2000.0, 5.75, 10.0)
        assert abs(z_large) < abs(z_small)


# ---------------------------------------------------------------------------
# estimate_fla
# ---------------------------------------------------------------------------

class TestEstimateFla:
    def test_known_value(self):
        # 300 HP, 480 V, eff=0.95, pf=0.90
        # P_out=223800 W → P_in=235579 W → S=261754 VA
        # FLA = 261754 / (√3 · 480) = 314.84 A
        fla = estimate_fla(300.0, 480.0, 0.95, 0.90)
        assert abs(fla - 314.84) < 0.5

    def test_scales_with_hp(self):
        fla_100 = estimate_fla(100.0, 480.0, 0.95, 0.90)
        fla_300 = estimate_fla(300.0, 480.0, 0.95, 0.90)
        assert abs(fla_300 / fla_100 - 3.0) < 0.01

    def test_higher_voltage_lower_current(self):
        fla_480 = estimate_fla(100.0, 480.0, 0.95, 0.90)
        fla_4160 = estimate_fla(100.0, 4160.0, 0.95, 0.90)
        assert fla_4160 < fla_480


# ---------------------------------------------------------------------------
# solve_load_voltage — spot check and startup-cap behaviour
# ---------------------------------------------------------------------------

class TestSolveLoadVoltage:
    def _z_th(self, s=BASE):
        z_xfmr = z_from_pctz(s.v_ll_hv, s.xfmr_kva, s.xfmr_pct_z, s.xfmr_x_over_r)
        z_line = complex(s.cond.r_ohm_per_mile * s.miles, s.cond.x_ohm_per_mile * s.miles)
        return z_xfmr + z_line

    def _vth(self, s=BASE):
        return complex(s.v_ll_hv / SQRT3, 0.0)

    def test_voltage_depressed_during_start(self):
        Z_th = self._z_th()
        Vth = self._vth()
        V_load = solve_load_voltage(BASE, Z_th, Vth)
        assert abs(V_load) < abs(Vth)

    def test_atl_const_i_sag_regression(self):
        Z_th = self._z_th()
        Vth = self._vth()
        V_load = solve_load_voltage(BASE, Z_th, Vth)
        sag = 100.0 * (1.0 - abs(V_load) / abs(Vth))
        assert abs(sag - SAG_BASE_PCT) < 0.05

    def test_startup_cap_reduces_sag(self):
        Z_th = self._z_th()
        Vth = self._vth()
        V_free = solve_load_voltage(BASE, Z_th, Vth)
        i_tight = (50.0 * 1000.0) / (SQRT3 * BASE.v_ll_hv)
        V_capped = solve_load_voltage(BASE, Z_th, Vth, i_start_limit_hv=i_tight)
        assert abs(V_capped) > abs(V_free)

    def test_startup_cap_reference(self):
        Z_th = self._z_th()
        Vth = self._vth()
        i_tight = (50.0 * 1000.0) / (SQRT3 * BASE.v_ll_hv)
        V_capped = solve_load_voltage(BASE, Z_th, Vth, i_start_limit_hv=i_tight)
        sag = 100.0 * (1.0 - abs(V_capped) / abs(Vth))
        assert abs(sag - 0.568) < 0.05


# ---------------------------------------------------------------------------
# run_return — end-to-end with property checks
# ---------------------------------------------------------------------------

class TestRunReturn:
    def test_atl_regression(self):
        sag, vlv = run_return(BASE)
        assert abs(sag - SAG_BASE_PCT) < 0.05

    def test_lv_voltage_consistent_with_sag(self):
        sag, vlv = run_return(BASE)
        expected_vlv = BASE.v_ll_lv * (1.0 - sag / 100.0)
        assert abs(vlv - expected_vlv) < 2.0

    def test_vfd_lower_sag_than_atl(self):
        sag_atl, _ = run_return(BASE)
        sag_vfd, _ = run_return(replace(BASE, start_mode="VFD"))
        assert sag_vfd < sag_atl

    def test_cap_bank_reduces_sag(self):
        sag_no, _ = run_return(BASE)
        sag_cap, _ = run_return(replace(BASE, cap_kvar=300.0))
        assert sag_cap < sag_no

    def test_line_length_increases_sag(self):
        sag_short, _ = run_return(replace(BASE, miles=1.0))
        sag_long, _ = run_return(replace(BASE, miles=10.0))
        assert sag_long > sag_short

    def test_upstream_z_increases_sag(self):
        sag_no, _ = run_return(BASE)
        sag_up, _ = run_return(BASE, z_upstream_phase=complex(0.5, 2.0))
        assert sag_up > sag_no

    def test_larger_motor_increases_sag(self):
        sag_small, _ = run_return(replace(BASE, hp=50.0))
        sag_large, _ = run_return(replace(BASE, hp=500.0))
        assert sag_large > sag_small

    def test_const_s_similar_to_const_i(self):
        sag_i, _ = run_return(replace(BASE, atl_model="CONST_I"))
        sag_s, _ = run_return(replace(BASE, atl_model="CONST_S"))
        assert abs(sag_i - sag_s) < 8.0  # same order of magnitude

    def test_vfd_sag_regression(self):
        sag, _ = run_return(replace(BASE, start_mode="VFD"))
        assert abs(sag - 0.536) < 0.05

    def test_cap_sag_regression(self):
        sag, _ = run_return(replace(BASE, cap_kvar=300.0))
        assert abs(sag - 3.200) < 0.05

    def test_line_sag_regression(self):
        sag, _ = run_return(replace(BASE, miles=5.0))
        assert abs(sag - 8.477) < 0.05
