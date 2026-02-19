import math
import cmath
from dataclasses import dataclass, field

SQRT3 = math.sqrt(3.0)

NEMA_KVA_PER_HP = {
    "G": (5.6, 6.29),
    "H": (6.3, 7.09),
    "J": (7.1, 7.99),
    "K": (8.0, 8.99),
}

def locked_rotor_kva(hp: float, code_letter: str) -> float:
    lo, hi = NEMA_KVA_PER_HP[code_letter.upper()]
    return hp * 0.5 * (lo + hi)

@dataclass(frozen=True)
class Conductor:
    r_ohm_per_mile: float
    x_ohm_per_mile: float

@dataclass
class Scenario:
    hp: float = 300.0
    motor_code: str = "G"

    v_ll_hv: float = 12_470.0
    v_ll_lv: float = 480.0

    miles: float = 7.0
    cond: Conductor = field(default_factory=lambda: Conductor(1.367, 0.581))  # #2 ACSR approx

    xfmr_kva: float = 500.0
    xfmr_pct_z: float = 5.75
    xfmr_x_over_r: float = 10.0

    # Start representation
    start_mode: str = "ATL"         # "ATL" or "VFD"
    atl_model: str = "CONST_I"      # "CONST_I" (recommended) or "CONST_S"
    start_pf: float = 0.25          # used only for CONST_S
    atl_I_multiplier_of_LRA: float = 1.0  # for CONST_I; 1.0 = LRA magnitude

    # VFD representation
    vfd_i_limit_pu_fla: float = 1.30
    run_eff: float = 0.95
    run_pf: float = 0.90

    # Capacitor bank (3φ total kVAr at the load bus)
    cap_kvar: float = 0.0

def z_from_pctz(v_ll: float, s_kva: float, pct_z: float, x_over_r: float) -> complex:
    z_base = (v_ll ** 2) / (s_kva * 1000.0)
    z_mag = (pct_z / 100.0) * z_base
    r = z_mag / math.sqrt(1.0 + x_over_r**2)
    x = r * x_over_r
    return complex(r, x)

def estimate_fla(hp: float, v_ll: float, eff: float, pf: float) -> float:
    p_out = hp * 746.0
    p_in = p_out / max(eff, 1e-9)
    s_va = p_in / max(pf, 1e-9)
    return s_va / (SQRT3 * v_ll)

def two_bus_solve_constS(Vth_phase: complex, Zth: complex, S3: complex, max_iter=200, tol=1e-6) -> complex:
    V = Vth_phase
    for _ in range(max_iter):
        I = (S3.conjugate() / (3.0 * V.conjugate()))
        V_new = Vth_phase - Zth * I
        if abs(V_new - V) / max(abs(V), 1e-9) < tol:
            return V_new
        V = V_new
    return V

def run(s: Scenario) -> None:
    # Build Thevenin impedance on HV side
    Z_line = complex(s.cond.r_ohm_per_mile * s.miles, s.cond.x_ohm_per_mile * s.miles)
    Z_xfmr = z_from_pctz(s.v_ll_hv, s.xfmr_kva, s.xfmr_pct_z, s.xfmr_x_over_r)
    Z_th = Z_line + Z_xfmr

    Vth_phase = complex(s.v_ll_hv / SQRT3, 0.0)

    # Capacitor bank input is 3-phase total kVAr at the load bus.
    Qcap_total_var = s.cap_kvar * 1000.0
    Qcap_phase_var = Qcap_total_var / 3.0

    if s.start_mode.upper() == "VFD":
        fla_lv = estimate_fla(s.hp, s.v_ll_lv, s.run_eff, s.run_pf)
        i_lim_lv = s.vfd_i_limit_pu_fla * fla_lv
        i_lim_hv = i_lim_lv * (s.v_ll_lv / s.v_ll_hv)

        # assume high displacement PF on front end; keep simple
        pf = 0.95
        I_motor = complex(i_lim_hv * pf, -i_lim_hv * math.sqrt(1.0 - pf*pf))  # lagging current
        # Capacitor adds leading reactive current: Icap = +j * Q/(3V)
        # We'll approximate by subtracting Q from motor's reactive demand at nominal V:
        Qm = abs(Vth_phase) * 3.0 * abs(I_motor) * math.sqrt(1.0 - pf*pf)  # crude
        Qnet = max(Qm - Qcap_total_var, 0.0)
        # rebuild I with same P but reduced Q (approx)
        Pm = abs(Vth_phase) * 3.0 * abs(I_motor) * pf
        Smag = math.sqrt(Pm*Pm + Qnet*Qnet)
        Imag = Smag / (3.0 * abs(Vth_phase))
        pf_new = Pm / max(Smag, 1e-9)
        I = complex(Imag * pf_new, -Imag * math.sqrt(max(1.0 - pf_new*pf_new, 0.0)))
        V_load = Vth_phase - Z_th * I

    else:
        # ATL
        S_lr_kva = locked_rotor_kva(s.hp, s.motor_code)

        if s.atl_model.upper() == "CONST_S":
            S = S_lr_kva * 1000.0
            P = S * s.start_pf
            Q = math.sqrt(max(S*S - P*P, 0.0))
            # subtract cap vars
            Q_net = Q - Qcap_total_var
            S3 = complex(P, Q_net)
            V_load = two_bus_solve_constS(Vth_phase, Z_th, S3)

        elif s.atl_model.upper() == "CONST_I":
            # Compute LRA current on HV side from locked-rotor kVA
            I_lra_hv = (S_lr_kva * 1000.0) / (SQRT3 * s.v_ll_hv)
            I_mag = s.atl_I_multiplier_of_LRA * I_lra_hv

            # Assume LRA PF (lagging). Start PF commonly 0.15–0.35; keep start_pf for angle.
            pf = s.start_pf
            I_motor = complex(I_mag * pf, -I_mag * math.sqrt(max(1.0 - pf*pf, 0.0)))

            # Capacitor current at the *load* bus depends on voltage: Icap = +j Qcap / (3 V*)
            # We'll iterate a couple times: V -> Icap -> V
            V = Vth_phase
            for _ in range(50):
                Icap = complex(0.0, +Qcap_phase_var / max(abs(V), 1e-9))  # leading (+j)
                I_total = I_motor + Icap  # net current drawn from source
                V_new = Vth_phase - Z_th * I_total
                if abs(V_new - V) / max(abs(V), 1e-9) < 1e-6:
                    V = V_new
                    break
                V = V_new

            V_load = V
        else:
            raise ValueError("atl_model must be CONST_S or CONST_I")

    sag_pct = 100.0 * (1.0 - abs(V_load) / abs(Vth_phase))
    v_lv_ll = abs(V_load) * SQRT3 * (s.v_ll_lv / s.v_ll_hv)

    print("=== Combined voltage sag (2-bus) ===")
    print(f"Mode: {s.start_mode.upper()}   (ATL model: {s.atl_model.upper() if s.start_mode.upper()=='ATL' else 'n/a'})")
    print(f"Cap bank: {s.cap_kvar:.0f} kVAr (3φ) at load bus")
    print(f"Z_th per phase: {Z_th.real:.2f} + j{Z_th.imag:.2f} Ω")
    print(f"HV sag: ~{sag_pct:.1f}%   |Vφ| {abs(Vth_phase):.0f} -> {abs(V_load):.0f} V")
    print(f"Approx LV during event: ~{v_lv_ll:.0f} V LL")
    print()

def run_return(s: Scenario) -> tuple[float, float]:
    # Build Thevenin impedance on HV side
    Z_line = complex(s.cond.r_ohm_per_mile * s.miles, s.cond.x_ohm_per_mile * s.miles)
    Z_xfmr = z_from_pctz(s.v_ll_hv, s.xfmr_kva, s.xfmr_pct_z, s.xfmr_x_over_r)
    Z_th = Z_line + Z_xfmr

    Vth_phase = complex(s.v_ll_hv / SQRT3, 0.0)

    # Capacitor bank input is 3-phase total kVAr at the load bus.
    Qcap_total_var = s.cap_kvar * 1000.0
    Qcap_phase_var = Qcap_total_var / 3.0

    if s.start_mode.upper() == "VFD":
        fla_lv = estimate_fla(s.hp, s.v_ll_lv, s.run_eff, s.run_pf)
        i_lim_lv = s.vfd_i_limit_pu_fla * fla_lv
        i_lim_hv = i_lim_lv * (s.v_ll_lv / s.v_ll_hv)

        # assume high displacement PF on front end; keep simple
        pf = 0.95
        I_motor = complex(i_lim_hv * pf, -i_lim_hv * math.sqrt(1.0 - pf*pf))  # lagging current
        # Capacitor adds leading reactive current: Icap = +j * Q/(3V)
        # We'll approximate by subtracting Q from motor's reactive demand at nominal V:
        Qm = abs(Vth_phase) * 3.0 * abs(I_motor) * math.sqrt(1.0 - pf*pf)  # crude
        Qnet = max(Qm - Qcap_total_var, 0.0)
        # rebuild I with same P but reduced Q (approx)
        Pm = abs(Vth_phase) * 3.0 * abs(I_motor) * pf
        Smag = math.sqrt(Pm*Pm + Qnet*Qnet)
        Imag = Smag / (3.0 * abs(Vth_phase))
        pf_new = Pm / max(Smag, 1e-9)
        I = complex(Imag * pf_new, -Imag * math.sqrt(max(1.0 - pf_new*pf_new, 0.0)))
        V_load = Vth_phase - Z_th * I

    else:
        # ATL
        S_lr_kva = locked_rotor_kva(s.hp, s.motor_code)

        if s.atl_model.upper() == "CONST_S":
            S = S_lr_kva * 1000.0
            P = S * s.start_pf
            Q = math.sqrt(max(S*S - P*P, 0.0))
            # subtract cap vars
            Q_net = Q - Qcap_total_var
            S3 = complex(P, Q_net)
            V_load = two_bus_solve_constS(Vth_phase, Z_th, S3)

        elif s.atl_model.upper() == "CONST_I":
            # Compute LRA current on HV side from locked-rotor kVA
            I_lra_hv = (S_lr_kva * 1000.0) / (SQRT3 * s.v_ll_hv)
            I_mag = s.atl_I_multiplier_of_LRA * I_lra_hv

            # Assume LRA PF (lagging). Start PF commonly 0.15–0.35; keep start_pf for angle.
            pf = s.start_pf
            I_motor = complex(I_mag * pf, -I_mag * math.sqrt(max(1.0 - pf*pf, 0.0)))

            # Capacitor current at the *load* bus depends on voltage: Icap = +j Qcap / (3 V*)
            # We'll iterate a couple times: V -> Icap -> V
            V = Vth_phase
            for _ in range(50):
                Icap = complex(0.0, +Qcap_phase_var / max(abs(V), 1e-9))  # leading (+j)
                I_total = I_motor + Icap  # net current drawn from source
                V_new = Vth_phase - Z_th * I_total
                if abs(V_new - V) / max(abs(V), 1e-9) < 1e-6:
                    V = V_new
                    break
                V = V_new

            V_load = V
        else:
            raise ValueError("atl_model must be CONST_S or CONST_I")

    sag_pct = 100.0 * (1.0 - abs(V_load) / abs(Vth_phase))
    v_lv_ll = abs(V_load) * SQRT3 * (s.v_ll_lv / s.v_ll_hv)

    return sag_pct, v_lv_ll
    pass

def trade_sweep():
    base = Scenario()
    base.start_mode = "ATL"
    base.atl_model = "CONST_I"
    base.start_pf = 0.25
    base.cap_kvar = 0.0

    # Candidate transformer sizes and %Z
    xfmr_kvas = [500, 750, 1000, 1500, 2000]
    pctzs = [4.5, 5.0, 5.75, 6.5]

    # Conductor options as (label, R, X)
    # Put real values here as you like. For now, include a couple rough upgrades.
    conductors = [
        ("#2 ACSR", 1.367, 0.581),
        ("1/0 ACSR", 0.86, 0.50),
        ("4/0 ACSR", 0.43, 0.45),
        ("336 ACSR", 0.27, 0.40),
    ]

    print("=== Trade sweep: transformer vs reconductor (ATL CONST_I) ===")
    print("Case | xfmr_kVA | %Z | conductor | sag% | LV_LL_V")
    print("-"*72)

    case = 1
    for label, r, x in conductors:
        for kva in xfmr_kvas:
            for pctz in pctzs:
                sc = Scenario(
                    hp=base.hp,
                    motor_code=base.motor_code,
                    v_ll_hv=base.v_ll_hv,
                    v_ll_lv=base.v_ll_lv,
                    miles=base.miles,
                    cond=Conductor(r, x),
                    xfmr_kva=kva,
                    xfmr_pct_z=pctz,
                    xfmr_x_over_r=base.xfmr_x_over_r,
                    start_mode=base.start_mode,
                    atl_model=base.atl_model,
                    start_pf=base.start_pf,
                    cap_kvar=base.cap_kvar
                )
                sag, vll = run_return(sc)
                print(f"{case:>4} | {kva:>8} | {pctz:>3.2f} | {label:<9} | {sag:>5.1f} | {vll:>7.0f}")
                case += 1

if __name__ == "__main__":
    trade_sweep()


if __name__ == "__main__":
    sc = Scenario()
    # Recommended: ATL as CONST_I for meaningful sags
    sc.start_mode = "ATL"
    sc.atl_model = "CONST_I"
    sc.start_pf = 0.25

    for cap in [0, 300, 600, 900, 1200, 1500]:
        sc.cap_kvar = cap
        run(sc)
