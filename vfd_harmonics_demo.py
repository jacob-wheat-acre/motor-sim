import math
import numpy as np
import matplotlib.pyplot as plt

PI = math.pi
SQRT3 = math.sqrt(3.0)


def thd_from_fft(mag, fundamental_idx):
    """THD = sqrt(sum(h>=2 of Ih^2)) / I1"""
    I1 = mag[fundamental_idx]
    if I1 <= 1e-12:
        return float("nan")
    harm_sq = np.sum(mag[2*fundamental_idx:]**2)  # rough (includes beyond)
    return math.sqrt(harm_sq) / I1


def harmonic_spectrum(i, fs, f1, max_h=50):
    """
    Returns harmonic orders h=1..max_h and RMS magnitudes of each harmonic.
    Uses single-sided FFT; maps bins nearest to h*f1.
    """
    n = len(i)
    i = i - np.mean(i)  # remove DC
    win = np.hanning(n)
    iw = i * win
    # FFT amplitude correction for Hann window RMS:
    # We'll use bin picking; relative magnitudes are what we want.
    I = np.fft.rfft(iw)
    freqs = np.fft.rfftfreq(n, d=1/fs)

    # magnitude scaling (approx): 2/N for single-sided, then /sqrt(2) for RMS
    mag = (2.0 / np.sum(win)) * np.abs(I)  # amplitude approx
    mag_rms = mag / math.sqrt(2.0)

    hs = np.arange(1, max_h + 1)
    mags = np.zeros_like(hs, dtype=float)
    for k, h in enumerate(hs):
        target = h * f1
        idx = int(np.argmin(np.abs(freqs - target)))
        mags[k] = mag_rms[idx]
    return hs, mags


def simulate_6pulse_rectifier(
    V_ll_rms=480.0,
    f=60.0,
    P_dc=150e3,         # DC load power (W) ~ inverter+motor mechanical demand (rough)
    C_dc=5e-3,          # DC link cap (F) 5 mF is plausible order
    Vdc_init=650.0,     # initial DC bus volts (for 480V LL, typical Vdc ~ 1.35*Vll = 648V)
    R_s=0.02,           # source resistance seen line-line (ohm)
    L_s=200e-6,         # source inductance seen line-line (H) (line reactor effect)
    V_diode=1.2,        # diode drop (V) per diode (2 diodes conduct)
    t_end=0.25,
    fs=20000.0,         # sim sample rate (Hz)
):
    """
    Very practical VFD front-end model:
      - 6-pulse bridge selects max phase as + and min phase as -
      - conduction happens only if v_ll_inst > Vdc + diode drops
      - source impedance modeled as series R-L on the conducting line-line path
      - DC link capacitor integrates current balance: C dV/dt = i_rect - i_load
      - load is constant power P_dc (clamped so it doesn't blow up at low Vdc)

    Outputs: time, ia, ib, ic, vdc
    """
    dt = 1.0 / fs
    n = int(t_end * fs)

    # Phase-to-neutral RMS (assuming balanced source)
    V_ph_rms = V_ll_rms / SQRT3
    V_ph_pk = V_ph_rms * math.sqrt(2.0)

    # time arrays
    t = np.arange(n) * dt

    # phase voltages (ideal source)
    wa = 2 * PI * f
    va = V_ph_pk * np.sin(wa * t)
    vb = V_ph_pk * np.sin(wa * t - 2 * PI / 3)
    vc = V_ph_pk * np.sin(wa * t + 2 * PI / 3)

    # states
    vdc = Vdc_init
    i_ll = 0.0  # conducting line-line current (through R-L path)

    # logs
    ia = np.zeros(n)
    ib = np.zeros(n)
    ic = np.zeros(n)
    vdc_log = np.zeros(n)

    # helper: constant power load current
    def i_load_from_vdc(v):
        v = max(v, 50.0)  # prevent singularity; below this is "drive undervoltage trip"
        return P_dc / v

    for k in range(n):
        # instantaneous phase voltages
        vph = np.array([va[k], vb[k], vc[k]])

        # select diode conduction pair (top = max, bottom = min)
        i_top = int(np.argmax(vph))
        i_bot = int(np.argmin(vph))

        v_ll_inst = vph[i_top] - vph[i_bot]  # ideal line-line driving voltage

        # diode drops (2 diodes conduct)
        v_drop = 2.0 * V_diode

        # conduction condition: source can charge DC bus
        v_drive = v_ll_inst - v_drop - vdc

        if v_drive > 0:
            # di/dt = (v_drive - R*i) / L
            di = (v_drive - R_s * i_ll) / max(L_s, 1e-9)
            i_ll = max(0.0, i_ll + dt * di)
        else:
            # not conducting: current decays to zero through source resistance (simple)
            # di/dt = -(R/L) i
            di = -(R_s / max(L_s, 1e-9)) * i_ll
            i_ll = max(0.0, i_ll + dt * di)

        # map the conducting line-line current to phase currents
        iph = np.zeros(3)
        iph[i_top] = +i_ll
        iph[i_bot] = -i_ll

        ia[k], ib[k], ic[k] = iph[0], iph[1], iph[2]

        # DC link current into capacitor is rectifier output current ≈ i_ll
        # (this is a simplification; it’s good enough for harmonic behavior)
        i_rect = i_ll
        i_load = i_load_from_vdc(vdc)

        dv = (i_rect - i_load) / max(C_dc, 1e-12)
        vdc = vdc + dt * dv
        vdc_log[k] = vdc

    return t, ia, ib, ic, vdc_log


def plot_time_and_spectrum(t, ia, f1=60.0, fs=20000.0, title=""):
    # show a few cycles at the end
    T = 1.0 / f1
    window = int(6 * T * fs)  # last ~6 cycles
    ia_seg = ia[-window:]
    t_seg = t[-window:] - t[-window]

    plt.figure()
    plt.plot(t_seg, ia_seg)
    plt.xlabel("Time (s)")
    plt.ylabel("Phase A current (A)")
    plt.title(f"{title} - phase current (last ~6 cycles)")
    plt.grid(True)

    # FFT on last 10 cycles for better resolution
    window2 = int(10 * T * fs)
    i_fft = ia[-window2:]
    hs, mags = harmonic_spectrum(i_fft, fs, f1, max_h=50)

    plt.figure()
    plt.stem(hs, mags, basefmt=" ")
    plt.xlabel("Harmonic order (h)")
    plt.ylabel("Ih RMS (A)")
    plt.title(f"{title} - harmonic spectrum (phase A)")
    plt.grid(True)

    # THD (using harmonic bins, exclude DC)
    I1 = mags[0]
    thd = math.sqrt(np.sum(mags[1:]**2)) / max(I1, 1e-12)
    print(f"{title}: I1={I1:.1f} A_rms, THD={100*thd:.1f}%")

    plt.show()


if __name__ == "__main__":
    fs = 20000.0
    f1 = 60.0

    # Baseline: small line reactor
    t, ia, ib, ic, vdc = simulate_6pulse_rectifier(
        V_ll_rms=480.0,
        f=f1,
        P_dc=200e3,       # try 50e3..250e3
        C_dc=8e-3,        # 8 mF
        Vdc_init=650.0,
        R_s=0.02,
        L_s=200e-6,       # "some" inductance
        t_end=0.25,
        fs=fs,
    )
    plot_time_and_spectrum(t, ia, f1=f1, fs=fs, title="6-pulse, Ls=200uH")

    # Compare: larger line reactor (lower THD)
    t2, ia2, _, _, _ = simulate_6pulse_rectifier(
        V_ll_rms=480.0,
        f=f1,
        P_dc=200e3,
        C_dc=8e-3,
        Vdc_init=650.0,
        R_s=0.02,
        L_s=2e-3,         # bigger reactor
        t_end=0.25,
        fs=fs,
    )
    plot_time_and_spectrum(t2, ia2, f1=f1, fs=fs, title="6-pulse, Ls=2mH")
