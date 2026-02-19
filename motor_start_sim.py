# motor_start_sim.py
import math
import cmath
from dataclasses import dataclass
import matplotlib.pyplot as plt

SQRT3 = math.sqrt(3.0)
PI = math.pi


# --------------------------
# Utilities
# --------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def current_phasor_from_pf(I_mag: float, pf: float) -> complex:
    """Lagging current phasor with magnitude I_mag and displacement PF=pf."""
    pf = clamp(pf, 0.0, 1.0)
    phi = math.acos(pf)
    return cmath.rect(I_mag, -phi)


def estimate_fla(hp: float, v_ll: float, eff: float = 0.95, pf: float = 0.90) -> float:
    """Crude full-load amps estimate. Good enough for VFD current-limit scaling."""
    P = hp * 746.0
    S = P / (eff * pf)
    return S / (SQRT3 * v_ll)


# --------------------------
# Induction motor model (steady-state equivalent circuit + "fake dynamics")
# --------------------------
@dataclass
class MotorParams:
    hp: float = 300.0
    poles: int = 4
    f_base: float = 60.0
    v_ll_base: float = 480.0

    # Per-phase equivalent circuit params (ohms) at base frequency, referred to stator
    R1: float = 0.015
    X1: float = 0.09
    R2: float = 0.015
    X2: float = 0.09
    Xm: float = 3.5  # magnetizing reactance

    # Mechanical inertia (kg*m^2)
    J: float = 25.0

    def w_sync(self, f_hz: float) -> float:
        """Synchronous mechanical speed (rad/s) at electrical frequency f_hz."""
        # w_sync = 2*pi*(120 f / poles)/60 = 4*pi*f/poles
        return (4.0 * PI * f_hz) / self.poles

    def rated_speed(self) -> float:
        return 0.97 * self.w_sync(self.f_base)

    def rated_torque(self) -> float:
        P = self.hp * 746.0
        return P / max(self.rated_speed(), 1e-9)


@dataclass
class FeederThevenin:
    """
    Thevenin source represented on the motor terminals (480 V side).
    z_th_phase is per-phase (ohms) on 480 V base.
    """
    v_ll: float = 480.0
    z_th_phase: complex = complex(0.010, 0.060)


@dataclass
class StartProfile:
    mode: str = "ATL"  # "ATL", "SOFT", "VFD"
    t_end: float = 8.0
    dt: float = 0.002

    # Soft starter voltage ramp
    soft_v_start_pu: float = 0.35
    soft_ramp_time: float = 3.0

    # VFD basic V/Hz ramp “shape” (used only for voltage command in VFD mode)
    vfd_vhz_boost_pu: float = 0.08  # low-speed voltage boost


@dataclass
class LoadParams:
    """
    CENTRIFUGAL: T_load = T_rated * (w / w_base)^2 (pump/fan)
    CONSTANT:    T_load = constant torque
    """
    kind: str = "CENTRIFUGAL"
    T_rated_pu: float = 1.0  # torque at rated speed relative to motor rated torque


@dataclass
class VFDControlParams:
    """
    Torque-controlled VFD block (simple FOC-like abstraction):
      - Speed reference ramp
      - PI -> torque command
      - Map torque -> Iq with fixed Id (flux current)
      - Enforce current limit
      - Compute feeder drop with assumed PF
      - Command V/Hz voltage from frequency
    """
    # Speed reference ramp (mechanical)
    w_ref_ramp_time: float = 4.0  # seconds to reach rated speed reference
    w_ref_min: float = 5.0        # rad/s (avoid exactly 0)

    # Speed PI -> torque command
    kp: float = 8.0
    ki: float = 25.0

    # Limits
    torque_limit_pu: float = 1.8
    i_limit_pu: float = 1.5

    # Flux-producing current fraction (Id) of FLA (simplified)
    i_flux_pu: float = 0.35

    # PF assumed at feeder (rectifier+dc-link tends to near-unity displacement PF)
    pf_assumed: float = 0.98

    # Minimum electrical frequency for stable V/Hz command
    f_min_hz: float = 5.0

    # Approx slip fraction at rated torque (used to estimate electrical frequency)
    slip_frac_at_rated_torque: float = 0.03


def motor_thevenin(V_phase: float, f: float, mp: MotorParams):
    """
    Thevenin equivalent seen by rotor branch.
    Reactances scale ~ f.
    """
    scale = f / mp.f_base
    X1 = mp.X1 * scale
    X2 = mp.X2 * scale
    Xm = mp.Xm * scale

    Z1 = complex(mp.R1, X1)
    Zm = complex(0.0, Xm)

    Vth = V_phase * (Zm / (Z1 + Zm))
    Zth = (Z1 * Zm) / (Z1 + Zm)
    return Vth, Zth, X2


def motor_torque_and_current(V_phase: float, f: float, slip: float, mp: MotorParams):
    """
    Steady-state equivalent circuit: returns (T_em [N*m], I_in [A] per-phase complex)
    Used for ATL/SOFT only.
    """
    slip = clamp(slip, 1e-4, 1.0)
    f = max(f, 1.0)  # avoid 0 Hz singularities; ATL/SOFT are at 60 anyway
    w_sync = mp.w_sync(f)

    Vth, Zth, X2 = motor_thevenin(V_phase, f, mp)

    Z2 = complex(mp.R2 / slip, X2)
    I2 = Vth / (Zth + Z2)

    P_ag = 3.0 * (abs(I2) ** 2) * (mp.R2 / slip)
    T_em = P_ag / max(w_sync, 1e-9)

    # Input current approximation (stator + magnetizing)
    scale = f / mp.f_base
    Z1 = complex(mp.R1, mp.X1 * scale)
    Zm = complex(0.0, mp.Xm * scale)

    V_node = V_phase - I2 * Z1
    I_mag = V_node / Zm
    I_in = I2 + I_mag

    return T_em, I_in


def load_torque(w: float, w_base: float, mp: MotorParams, lp: LoadParams) -> float:
    """Mechanical load torque (N*m)."""
    T_rated = lp.T_rated_pu * mp.rated_torque()

    kind = lp.kind.upper()
    if kind == "CENTRIFUGAL":
        pu = (w / max(w_base, 1e-9)) ** 2
        return T_rated * pu
    elif kind == "CONSTANT":
        return T_rated
    else:
        # fallback
        return T_rated


def start_voltage_and_freq(t: float, sp: StartProfile, mp: MotorParams):
    """
    Source-side command (before Thevenin drop).

    ATL:  V=1.0 pu, f=60
    SOFT: V ramps, f=60
    VFD:  (in this file) the VFD torque-control block determines f_cmd & V_cmd;
          this function is NOT used for VFD mode.
    """
    mode = sp.mode.upper()

    if mode == "ATL":
        return mp.v_ll_base, mp.f_base

    if mode == "SOFT":
        if t <= sp.soft_ramp_time:
            a = t / max(sp.soft_ramp_time, 1e-9)
            v_pu = sp.soft_v_start_pu + a * (1.0 - sp.soft_v_start_pu)
        else:
            v_pu = 1.0
        return mp.v_ll_base * v_pu, mp.f_base

    if mode == "VFD":
        raise ValueError("start_voltage_and_freq() not used in VFD mode (handled in VFD block).")

    raise ValueError("Unknown start mode")


def simulate(mp: MotorParams, feeder: FeederThevenin, sp: StartProfile, lp: LoadParams, vfd: VFDControlParams | None = None):
    dt = sp.dt
    n = int(sp.t_end / dt) + 1

    # Base speed references
    w_sync_base = mp.w_sync(mp.f_base)
    w_rated = mp.rated_speed()
    w_base = w_rated

    # State
    w = 0.0  # mechanical rad/s

    # Logs
    t_log, w_log, slip_log = [], [], []
    vll_log, ill_log, torque_log, tload_log, f_log = [], [], [], [], []

    # Common scalars
    I_fla = estimate_fla(mp.hp, mp.v_ll_base)
    T_rated = mp.rated_torque()

    # VFD params & derived limits
    if vfd is None:
        vfd = VFDControlParams()

    I_limit_vfd = vfd.i_limit_pu * I_fla
    T_limit_vfd = vfd.torque_limit_pu * T_rated

    # A simple torque constant mapping for the VFD abstraction:
    # Kt ≈ T_rated / I_rated (N*m per A). This is "FOC-ish", not nameplate-precise.
    Kt = T_rated / max(I_fla, 1e-9)

    # PI integrator
    int_err = 0.0

    for k in range(n):
        t = k * dt

        # -----------------------------
        # VFD torque-controlled block
        # -----------------------------
        if sp.mode.upper() == "VFD":
            # 1) Speed reference ramp
            a = clamp(t / max(vfd.w_ref_ramp_time, 1e-9), 0.0, 1.0)
            w_ref = max(vfd.w_ref_min, a * w_rated)

            # 2) Load torque at current speed
            T_ld = load_torque(w, w_base, mp, lp)

            # 3) PI speed control -> torque command
            err = w_ref - w
            int_err += err * dt

            T_cmd = vfd.kp * err + vfd.ki * int_err
            # For a pump load in motoring direction, clamp at 0..T_limit
            T_cmd = clamp(T_cmd, 0.0, T_limit_vfd)

            # 4) Torque -> current demand (FOC-like: fixed Id, torque via Iq)
            Id = vfd.i_flux_pu * I_fla
            Iq = T_cmd / max(Kt, 1e-9)
            I_mag = math.sqrt(Id * Id + Iq * Iq)

            # Enforce current limit (scale Id/Iq together)
            if I_mag > I_limit_vfd:
                scale = I_limit_vfd / I_mag
                Id *= scale
                Iq *= scale
                I_mag = I_limit_vfd
                T_cmd = Kt * Iq  # torque reduces with Iq

            # 5) Estimate commanded electrical frequency
            # We need electrical frequency to set inverter voltage (V/Hz).
            # Approximate: w_sync ≈ w / (1 - s), with slip s based on torque fraction.
            torque_frac = T_cmd / max(T_rated, 1e-9)
            s_est = clamp(vfd.slip_frac_at_rated_torque * torque_frac, 0.0, 0.15)
            w_sync_est = w / max(1.0 - s_est, 1e-6)

            f_cmd = (w_sync_est * mp.poles) / (4.0 * PI)
            f_cmd = max(f_cmd, vfd.f_min_hz)

            # 6) Command voltage (V/Hz + low-speed boost)
            v_pu = clamp(f_cmd / mp.f_base, 0.0, 1.0)
            v_pu = v_pu + sp.vfd_vhz_boost_pu * (1.0 - v_pu)
            v_pu = clamp(v_pu, 0.0, 1.0)

            Vsrc_ll = mp.v_ll_base * v_pu
            Vsrc_ph = Vsrc_ll / SQRT3

            # 7) Feeder drop using assumed PF
            I_ph = current_phasor_from_pf(I_mag, vfd.pf_assumed)
            Vt = complex(Vsrc_ph, 0.0) - feeder.z_th_phase * I_ph

            Vt_ll = abs(Vt) * SQRT3

            # 8) Mechanical dynamics
            dw = (T_cmd - T_ld) / max(mp.J, 1e-9)
            w = max(0.0, w + dt * dw)

            # 9) Logs
            t_log.append(t)
            w_log.append(w)
            slip_log.append(0.0)  # not used in VFD block
            vll_log.append(Vt_ll)
            ill_log.append(I_mag)
            torque_log.append(T_cmd)
            tload_log.append(T_ld)
            f_log.append(f_cmd)

            continue  # IMPORTANT: do not run induction-slip model in VFD mode

        # -----------------------------
        # ATL / SOFT: induction model + Thevenin iteration
        # -----------------------------
        Vsrc_ll, f_cmd = start_voltage_and_freq(t, sp, mp)
        Vsrc_ph = Vsrc_ll / SQRT3

        # Synchronous speed & slip
        w_sync = mp.w_sync(f_cmd)
        slip = clamp((w_sync - w) / max(w_sync, 1e-9), 0.0, 1.0)

        # Solve motor terminal voltage: Vt = Vsrc - Zth * I(Vt)
        Vt = complex(Vsrc_ph, 0.0)
        for _ in range(30):
            T_em, I_in = motor_torque_and_current(abs(Vt), f_cmd, slip, mp)
            V_new = complex(Vsrc_ph, 0.0) - feeder.z_th_phase * I_in

            if abs(V_new - Vt) / max(abs(Vt), 1e-9) < 1e-6:
                Vt = V_new
                break
            Vt = V_new

        # Final electrical quantities at converged Vt
        T_em, I_in = motor_torque_and_current(abs(Vt), f_cmd, slip, mp)

        Vt_ll = abs(Vt) * SQRT3
        I_line = abs(I_in)  # approx

        # Mechanical load torque
        T_ld = load_torque(w, w_base, mp, lp)

        # Dynamics
        dw = (T_em - T_ld) / max(mp.J, 1e-9)
        w = max(0.0, w + dt * dw)

        # Logs
        t_log.append(t)
        w_log.append(w)
        slip_log.append(slip)
        vll_log.append(Vt_ll)
        ill_log.append(I_line)
        torque_log.append(T_em)
        tload_log.append(T_ld)
        f_log.append(f_cmd)

    return {
        "t": t_log,
        "w": w_log,
        "slip": slip_log,
        "vll": vll_log,
        "I": ill_log,
        "Tem": torque_log,
        "Tload": tload_log,
        "f": f_log,
    }


def plot_results(results, title):
    t = results["t"]

    plt.figure()
    plt.plot(t, results["vll"])
    plt.xlabel("Time (s)")
    plt.ylabel("Motor terminal voltage (V LL)")
    plt.title(f"{title} - Voltage")
    plt.grid(True)

    plt.figure()
    plt.plot(t, results["I"])
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A, approx)")
    plt.title(f"{title} - Current")
    plt.grid(True)

    plt.figure()
    plt.plot(t, results["w"])
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (rad/s)")
    plt.title(f"{title} - Speed")
    plt.grid(True)

    plt.figure()
    plt.plot(t, results["Tem"], label="Motor torque")
    plt.plot(t, results["Tload"], label="Load torque")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N·m)")
    plt.title(f"{title} - Torque")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    mp = MotorParams()

    # "Weak-ish" feeder on 480 V side (tune to taste)
    feeder = FeederThevenin(
        v_ll=mp.v_ll_base,
        z_th_phase=complex(0.010, 0.060)
    )

    # Change to LoadParams(kind="CONSTANT") for constant-torque load
    lp = LoadParams(kind="CENTRIFUGAL", T_rated_pu=1.0)

    vfd_ctrl = VFDControlParams(
        w_ref_ramp_time=4.0,
        kp=8.0,
        ki=25.0,
        torque_limit_pu=1.8,
        i_limit_pu=1.5,
        i_flux_pu=0.35,
        pf_assumed=0.98,
        f_min_hz=5.0,
        slip_frac_at_rated_torque=0.03
    )

    for mode in ["ATL", "SOFT", "VFD"]:
        sp = StartProfile(mode=mode)
        res = simulate(mp, feeder, sp, lp, vfd=vfd_ctrl)
        plot_results(res, f"{mode} start ({lp.kind.lower()} load)")
