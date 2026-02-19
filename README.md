# Motor Simulation App

This repo simulates voltage sag and power-quality effects during motor starting and VFD operation.

## What it does

- Models feeder + transformer Thevenin impedance and estimates start-event voltage sag.
- Compares across:
  - motor size (HP)
  - feeder length (miles)
  - transformer size
  - starting method (`ATL`, `SoftStart`, `VFD`)
  - conductor type (ACSR)
- Produces heatmaps and risk contours against a rapid-voltage-change limit.

Main files:

- `/Users/jacob/Code/motor-sim/volt_sag.py`: core 2-bus sag model and scenario definitions
- `/Users/jacob/Code/motor-sim/motor_flicker_sweep.py`: HP/miles sweeps with two focused plotting modes
- `/Users/jacob/Code/motor-sim/motor_start_sim.py`: dynamic-ish motor startup model
- `/Users/jacob/Code/motor-sim/vfd_harmonics_demo.py`: 6-pulse front-end harmonic demo

## Conductor impedance values used

For conductor comparison, `motor_flicker_sweep.py` now includes:

- `795 ACSR`: `R = 0.119 ohm/mi`, `X = 0.412 ohm/mi`
- `336 ACSR`: `R = 0.276 ohm/mi`, `X = 0.463 ohm/mi`
- `2/0 ACSR`: `R = 0.686 ohm/mi`, `X = 0.537 ohm/mi`
- `#2 ACSR`: `R = 1.367 ohm/mi`, `X = 0.581 ohm/mi`
- `#4 ACSR`: `R = 2.175 ohm/mi`, `X = 0.608 ohm/mi`

Basis:

- Converted from standard ACSR table values in `ohm/kft` using:
  - `ohm/mi = ohm/kft * 5.28`
- Resistance basis: AC resistance at 25C
- Reactance basis: 60 Hz inductive reactance at 1 ft equivalent spacing

Note: line reactance depends on geometry/spacing. If your feeder spacing differs, keep `R` and update `X` accordingly.

## Engineering model and equations

This project uses a practical 2-bus Thevenin model at the MV side to estimate start-event voltage sag.

Capacitor-bank convention used everywhere in this repo:

- User input `cap_kvar` is always **3-phase total kVAr**.
- Internal conversion:
  - `Q_cap,total,var = cap_kvar * 1000`
  - `Q_cap,phase,var = Q_cap,total,var / 3`

### 1) System Thevenin impedance

Per-phase source impedance seen by the motor bus:

- `Z_th = Z_line + Z_xfmr`
- `Z_line = miles * (R_cond + j X_cond)`

Transformer impedance from nameplate `%Z` and `X/R`:

- `Z_base = V_LL^2 / S_3phi`
- `|Z_xfmr| = (%Z/100) * Z_base`
- `R_xfmr = |Z_xfmr| / sqrt(1 + (X/R)^2)`
- `X_xfmr = R_xfmr * (X/R)`
- `Z_xfmr = R_xfmr + j X_xfmr`

Where:
- `V_LL` is MV line-line voltage (V)
- `S_3phi` is transformer kVA converted to VA

The Thevenin phase voltage is:

- `V_th,ph = V_LL / sqrt(3)`

### 2) Load-bus voltage during start

Core phasor equation:

- `V_load = V_th - Z_th * I_total`

The model computes `I_total` differently by start technology (below).  
Voltage sag reported:

- `Sag_% = 100 * (1 - |V_load|/|V_th|)`

Approximate LV line-line voltage during the event:

- `V_LV,event = |V_load| * sqrt(3) * (V_LV_base / V_HV_base)`

### 3) Locked-rotor and motor current foundations (ATL/SoftStart path)

Locked-rotor kVA uses NEMA code-letter bands:

- `S_LR,kVA = HP * midpoint(kVA/HP band)`

For constant-current ATL modeling:

- `I_LRA,HV = (S_LR * 1000) / (sqrt(3) * V_HV,LL)`
- `I_motor = k_start * I_LRA,HV * (pf - j*sqrt(1-pf^2))`

Where:
- `k_start = atl_I_multiplier_of_LRA` (`1.0` for ATL, lower for soft-start)
- `pf` is start power factor (lagging, default low)

If a capacitor bank is included, capacitor current is voltage-dependent and solved iteratively:

- `I_cap = +j * Q_cap,phase,var / |V|`
- `I_total = I_motor + I_cap`
- iterate `V_new = V_th - Z_th * I_total` to convergence

### 4) VFD path

VFD start is represented as current-limited with near-unity displacement PF:

- `FLA_LV = P_out/(eff*pf_run) / (sqrt(3)*V_LV,LL)` with `P_out = HP*746`
- `I_lim,LV = (vfd_i_limit_pu_fla) * FLA_LV`
- `I_lim,HV = I_lim,LV * (V_LV,LL / V_HV,LL)`
- `I_motor ~= I_lim,HV * (pf_vfd - j*sqrt(1-pf_vfd^2))`, with `pf_vfd ~ 0.95`

Capacitor interaction is approximated by reducing net reactive demand before rebuilding current magnitude.

### 5) Flicker screening thresholds used

Allowable rapid voltage-change limit is chosen from start frequency (`starts/hour`) using the MV table logic in code.  
Two thresholds are used for screening:

- `limit_threshold = allowable_dV%`
- `review_threshold = 0.8 * limit_threshold`

Engineering screening rule:

- If requested motor HP exceeds `Max HP (No Review)` at the given conductor/miles/start technology, trigger detailed flicker review.

## Starting technologies and impact on equations

### ATL (Across-the-Line)

What it is:
- Motor connected directly to the line at full voltage/frequency.

How it impacts equations:
- Uses highest inrush representation (`k_start = 1.0`).
- `I_total` is dominated by locked-rotor current, so `Z_th * I_total` is large.
- Produces the highest predicted sag for a given feeder stiffness.

### SoftStart

What it is:
- Solid-state starter that reduces applied voltage/current during acceleration.

How it impacts equations in this model:
- Implemented as ATL-style current model with reduced inrush multiplier (`k_start < 1`; currently `0.5`).
- Same network equations, but lower `I_motor` magnitude.
- Reduces sag relative to ATL; amount depends on selected multiplier and PF assumptions.

### VFD (Variable Frequency Drive)

What it is:
- Rectifier + DC bus + inverter; starts by controlling frequency/voltage and limiting current.

How it impacts equations:
- Replaces locked-rotor current model with current-limited input model tied to FLA.
- Uses higher assumed front-end displacement PF than ATL start.
- Usually gives the lowest sag for the same motor HP and feeder.

### Practical interpretation

- Engineering sensitivity is mostly through `Z_th` (feeder miles, conductor, transformer stiffness) and `|I_total|` (start method).
- ATL and weak feeders push points into review/violation zones quickly.
- VFD often shifts the same point back into `OK`, which is why it is a common flicker mitigation path.

## Run

From `/Users/jacob/Code/motor-sim`:

```bash
python3 motor_flicker_sweep.py
python3 volt_sag.py
python3 motor_start_sim.py
python3 vfd_harmonics_demo.py
```

`motor_flicker_sweep.py` also accepts user inputs for screening:

```bash
python3 motor_flicker_sweep.py --motor-hp 200 --miles 7 --start-tech ATL
python3 motor_flicker_sweep.py --motor-hp 200 --miles 7 --start-tech SoftStart
python3 motor_flicker_sweep.py --motor-hp 200 --miles 7 --start-tech VFD
python3 motor_flicker_sweep.py --motor-hp 200 --miles 7 --start-tech ATL --cap-kvar 300
```

Interactive prompt mode:

```bash
python3 motor_flicker_sweep.py --prompt
```

In prompt mode, the script asks:
- motor size (HP)
- feeder miles
- starting technology
- whether a nearby capacitor bank is installed (`yes/no`)
- capacitor-bank size in kVAr if installed

## How to make the charts less overwhelming

Implemented display pattern in `motor_flicker_sweep.py`:

1. Keep one primary decision chart.
- `overview_mode()`: one heatmap (`HP vs miles`) for a selected scenario.
- Includes risk overlay bands and contours:
  - `Green`: below review threshold
  - `Yellow`: review threshold to limit
  - `Red`: above limit

2. Add one compact comparison chart.
- `compare_mode()`: one selected operating point and conductor comparison.
- Shows:
  - horizontal bar chart of sag by conductor
  - ranked table with sag, margin, status, and screening HP thresholds

3. Use a practical screening heuristic.
- `review_threshold = 0.8 * flicker_limit`
- `limit_threshold = flicker_limit`
- If requested motor HP exceeds `Max HP (No Review)`, trigger detailed engineering flicker review.
