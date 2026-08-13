"""
Robustness check for the H2 / proximity_ecol scenario in predict_scenarios.py.

Approach A: direct utility contrast from the theta_ecol posterior, computed
in closed form instead of via pm.sample_posterior_predictive. Cross-checking
against predict_scenarios.py's proximity_ecol scenario guards against
mistakes in the general scenario engine's utility-building code.

For a hypothetical individual with high vs. low ecological orientation
(all other value dimensions held at population mean = 0), computes:

  ΔU(domestic, prox)  = θ_ecol[prox] × Δz
  ΔU(foreign,  prox)  = (θ_ecol[prox] + θ_ecol[source_purpose_foreign]) × Δz

No new model fitting — ΔU is directly readable from the existing posterior.

Standalone script, not wired into the Snakefile.
Run: python scripts/postprocessing/robustness/predict_approach_a.py
Output: output/predictions/prediction_approach_a.txt
"""
import os
import numpy as np
import arviz as az
from datetime import datetime

idata_path = "output/data/inference_main_hybrid_choice.nc"
z_high     = 1.5   # SD units above population mean on latent ecological scale
z_low      = -1.5  # SD units below population mean
out_path   = "output/predictions/prediction_approach_a.txt"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

# ---- load posterior ----

print("Loading posterior...")
posterior  = az.from_netcdf(idata_path).posterior
theta_ecol = posterior["theta_ecol"]          # (chain, draw, level)
level_coords = [str(l) for l in theta_ecol.level.values]
delta_z = z_high - z_low


# ---- helpers ----

def hdi_1d(arr, prob=0.89):
    arr = np.sort(arr)
    n   = len(arr)
    k   = int(np.floor(prob * n))
    widths = arr[k:] - arr[:n - k]
    i = int(np.argmin(widths))
    return float(arr[i]), float(arr[i + k])


def block(draws, label):
    draws = np.asarray(draws).ravel()
    med   = float(np.median(draws))
    lo, hi = hdi_1d(draws)
    p_neg = float(np.mean(draws < 0))
    p_pos = float(np.mean(draws > 0))
    return (
        f"  {label}\n"
        f"    Median ΔU : {med:+.3f}\n"
        f"    89% HDI   : [{lo:+.3f}, {hi:+.3f}]\n"
        f"    P(ΔU < 0) : {p_neg:.3f}   P(ΔU > 0) : {p_pos:.3f}"
    )


# ---- proximity level ordering ----

prox_levels = [
    "attr_vicinity_abroad",
    "attr_vicinity_another region",
    "attr_vicinity_your region",
    "attr_vicinity_your municipality",
]
prox_labels = {
    "attr_vicinity_abroad":            "Abroad",
    "attr_vicinity_another region":    "Another region",
    "attr_vicinity_your region":       "Your region",
    "attr_vicinity_your municipality": "Your municipality",
}

source_l      = "attr_source_purpose_foreign"
theta_source  = theta_ecol.sel(level=source_l).values.ravel()


# ---- build output ----

L = []

def hr(char="-"): L.append(char * 70)
def h1(text):     hr("="); L.append(text); hr("=")
def h2(text):     hr("-"); L.append(text); hr("-")
def blank():      L.append("")

h1("Prediction Approach A: utility contrast by ecological orientation")
L.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
L.append(f"Posterior : {idata_path}")
blank()

L.append("Settings")
L.append(f"  z_high  = {z_high}  (latent SD units above population mean)")
L.append(f"  z_low   = {z_low}  (population mean, centred to 0 in the model)")
L.append(f"  Δz      = {delta_z}")
L.append(f"  Socio-economic and socio-cultural values held at population mean.")
blank()

L.append("Method")
L.append("  ΔU is the shift in utility attributable solely to the ecological value")
L.append("  difference; country-specific effects (beta + gamma + interaction terms)")
L.append("  are identical for both ecological profiles and cancel in the contrast.")
L.append("  Baseline levels (Abroad / Domestic) have θ_ecol = 0 by construction")
L.append("  under reference-level coding, so their domestic ΔU = 0.")
blank()

# ---- domestic CO2 ----

h2("Domestic CO2  |  ΔU = θ_ecol[prox] × Δz")
blank()

for lvl in prox_levels:
    label = prox_labels[lvl]
    if lvl not in level_coords:
        L.append(f"  {label}")
        L.append(f"    ΔU = 0 by construction (reference level)")
    else:
        theta_p = theta_ecol.sel(level=lvl).values.ravel()
        L.append(block(theta_p * delta_z, label))
    blank()

# ---- foreign CO2 ----

h2("Foreign CO2  |  ΔU = (θ_ecol[prox] + θ_ecol[source/purpose]) × Δz")
blank()
L.append("  Note: for Abroad (baseline proximity), only the source/purpose")
L.append("  moderation term contributes (θ_ecol[abroad] = 0 by construction).")
blank()

for lvl in prox_levels:
    label = prox_labels[lvl]
    if lvl not in level_coords:
        L.append(block(theta_source * delta_z, f"{label}  (source/purpose term only)"))
    else:
        theta_p = theta_ecol.sel(level=lvl).values.ravel()
        L.append(block((theta_p + theta_source) * delta_z, label))
    blank()

# ---- domestic vs foreign contrast at your municipality ----

h2("Domestic vs. foreign contrast at Your municipality")
blank()
L.append("  How much does ecological orientation additionally shift utility")
L.append("  under foreign CO2 relative to domestic CO2?")
L.append("  = θ_ecol[source/purpose] × Δz  (interaction contrast)")
blank()

theta_mun = theta_ecol.sel(level="attr_vicinity_your municipality").values.ravel()
du_dom    = theta_mun * delta_z
du_for    = (theta_mun + theta_source) * delta_z
L.append(block(du_dom,            "Domestic CO2, your municipality"))
blank()
L.append(block(du_for,            "Foreign CO2,  your municipality"))
blank()
L.append(block(du_for - du_dom,   "Extra shift under foreign CO2 (= source/purpose term)"))
blank()

# ---- raw theta summaries for reference ----

h2("Raw theta_ecol posteriors for reference levels")
blank()
L.append("  θ_ecol[your municipality]")
theta_mun_vals = theta_ecol.sel(level="attr_vicinity_your municipality").values.ravel()
lo, hi = hdi_1d(theta_mun_vals)
L.append(f"    Median: {np.median(theta_mun_vals):+.3f}   89% HDI: [{lo:+.3f}, {hi:+.3f}]")
blank()
L.append("  θ_ecol[source/purpose foreign]")
lo, hi = hdi_1d(theta_source)
L.append(f"    Median: {np.median(theta_source):+.3f}   89% HDI: [{lo:+.3f}, {hi:+.3f}]")
blank()
hr("=")

output = "\n".join(L)
print(output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nSaved to {out_path}")
