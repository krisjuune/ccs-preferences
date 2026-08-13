"""
Robustness check for the H2 / proximity_ecol scenario in predict_scenarios.py.

Approaches B and C: absolute choice probabilities from the existing posterior,
computed manually instead of via pm.sample_posterior_predictive. Cross-checking
against predict_scenarios.py's proximity_ecol scenario guards against
mistakes in the general scenario engine's utility-building code.

Scenario: Option A (project in your municipality) vs Option B (project abroad),
all other attributes at their reference levels.

Approach B: fix z directly (SD units from population mean), compute
  P(choose A over B | z) = sigmoid(ΔU(z))  per posterior draw.

Approach C: specify ecological-value Likert item responses (y1, y2, y3),
  derive the conjugate Gaussian posterior for z per draw analytically, then
  sample z* from that posterior and propagate through the utility expression.

The key difference from Approach A: instead of reporting ΔU, we report
P(choose A), so results are on a [0, 1] probability scale.

Standalone script, not wired into the Snakefile.
Run: python scripts/postprocessing/robustness/predict_approach_bc.py
Output: output/predictions/prediction_approach_bc.txt
"""
import os
import numpy as np
import arviz as az
from datetime import datetime

idata_path = "output/data/inference_main_hybrid_choice.nc"
out_path   = "output/predictions/prediction_approach_bc.txt"

# ---- Approach B settings ----
z_high = 1.5   # SD units above population mean on latent ecological scale
z_low  = -1.5  # SD units below population mean

# ---- Approach C settings ----
# Ecological value Likert items (socio_ecological_1/2/3), same scale as training data.
y_high = np.array([5.0, 5.0, 5.0])   # maximum ecological orientation
y_low  = np.array([1.0, 1.0, 1.0])   # minimum ecological orientation

# ---- model parameter coordinates ----
PROX_LEVEL    = "attr_vicinity_your municipality"
SOURCE_LEVEL  = "attr_source_purpose_foreign"
INTERACT_NAME = "attr_vicinity_your municipality::attr_source_purpose_foreign"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

# ---- load posterior ----

print("Loading posterior...")
posterior = az.from_netcdf(idata_path).posterior

beta           = posterior["beta"]
gamma          = posterior["gamma"]
theta_ecol     = posterior["theta_ecol"]
beta_interact  = posterior["beta_interact"]
gamma_interact = posterior["gamma_interact"]
lambda_ecol    = posterior["lambda_ecol"]   # (chain, draw, 2): loadings for items 2 and 3
likert_sigma   = posterior["likert_sigma"]  # (chain, draw)

countries = [str(c) for c in gamma.country.values]

# Flatten to 1-D arrays of length n_chains * n_draws
beta_mun  = beta.sel(level=PROX_LEVEL).values.ravel()
beta_int  = beta_interact.sel(prox_source_interact=INTERACT_NAME).values.ravel()
theta_mun = theta_ecol.sel(level=PROX_LEVEL).values.ravel()

gamma_mun = {c: gamma.sel(country=c, level=PROX_LEVEL).values.ravel()              for c in countries}
gamma_int = {c: gamma_interact.sel(country=c, prox_source_interact=INTERACT_NAME).values.ravel() for c in countries}

lam1 = lambda_ecol[..., 0].values.ravel()   # loading for item 2
lam2 = lambda_ecol[..., 1].values.ravel()   # loading for item 3
sig  = likert_sigma.values.ravel()


# ---- helpers ----

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hdi_1d(arr, prob=0.89):
    arr = np.sort(arr)
    n   = len(arr)
    k   = int(np.floor(prob * n))
    widths = arr[k:] - arr[:n - k]
    i = int(np.argmin(widths))
    return float(arr[i]), float(arr[i + k])


def delta_u_domestic(z, country):
    """ΔU for Option A (your municipality, domestic) vs Option B (abroad, domestic)."""
    return beta_mun + gamma_mun[country] + theta_mun * z


def delta_u_foreign(z, country):
    """ΔU for Option A (your municipality, foreign) vs Option B (abroad, foreign).

    The β[foreign] and γ[c, foreign] terms cancel because both options share
    the same framing, so only the proximity + interaction terms remain.
    """
    return beta_mun + gamma_mun[country] + beta_int + gamma_int[country] + theta_mun * z


def z_from_items(y):
    """Sample z* from its conjugate Gaussian posterior given item responses y = [y1, y2, y3].

    Prior:       z ~ N(0, 1)
    Likelihood:  y1 ~ N(z, σ),  y2 ~ N(λ1 z, σ),  y3 ~ N(λ2 z, σ)
    Posterior:   z* | y ~ N(μ*, s*) where
      μ* = (y1 + λ1 y2 + λ2 y3) / (σ² + 1 + λ1² + λ2²)
      s* = σ / √(σ² + 1 + λ1² + λ2²)

    Returns one z* sample per posterior draw (array of length n_chains * n_draws).
    """
    y1, y2, y3 = float(y[0]), float(y[1]), float(y[2])
    denom  = sig**2 + 1.0 + lam1**2 + lam2**2
    mu_star = (y1 + lam1 * y2 + lam2 * y3) / denom
    s_star  = sig / np.sqrt(denom)
    return np.random.normal(mu_star, s_star)


def z_moments(y):
    """Return (mu_star, s_star) arrays without sampling, for reporting."""
    y1, y2, y3 = float(y[0]), float(y[1]), float(y[2])
    denom   = sig**2 + 1.0 + lam1**2 + lam2**2
    mu_star = (y1 + lam1 * y2 + lam2 * y3) / denom
    s_star  = sig / np.sqrt(denom)
    return mu_star, s_star


# ---- output formatting ----

L = []

def hr(char="-"): L.append(char * 70)
def h1(text):     hr("="); L.append(text); hr("=")
def h2(text):     hr("-"); L.append(text); hr("-")
def blank():      L.append("")


def fmt_p(probs, label):
    med    = float(np.median(probs))
    lo, hi = hdi_1d(probs)
    return f"    {label:<32s}  median: {med:.3f}  89% HDI: [{lo:.3f}, {hi:.3f}]"


def fmt_dp(dp, label):
    med    = float(np.median(dp))
    lo, hi = hdi_1d(dp)
    p_pos  = float(np.mean(dp > 0))
    return (
        f"    {label:<32s}  median: {med:+.3f}  89% HDI: [{lo:+.3f}, {hi:+.3f}]"
        f"  P(ΔP > 0): {p_pos:.3f}"
    )


# ---- header ----

h1("Prediction Approaches B & C: choice probabilities by ecological orientation")
L.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
L.append(f"Posterior : {idata_path}")
blank()
L.append("Scenario")
L.append("  Option A: project in your municipality  vs  Option B: project abroad.")
L.append("  All other attributes held at their reference levels.")
L.append("  CO₂ framing is held constant across both options (domestic or foreign),")
L.append("  so the shared framing term cancels in ΔU and only proximity drives P.")
blank()
L.append("  P   = sigmoid(ΔU)  = P(choose your municipality over abroad)")
L.append("  ΔP  = P(high ecological orientation) − P(low ecological orientation)")
blank()

# ---- Approach B ----

h2(f"Approach B  |  z fixed at ±{z_high} SD from population mean")
blank()
L.append(f"  z_high = {z_high}  (above population mean)")
L.append(f"  z_low  = {z_low}  (below population mean)")
blank()

for country in countries:
    L.append(f"  {country.capitalize()}")
    blank()

    du_dom_hi = delta_u_domestic(z_high, country)
    du_dom_lo = delta_u_domestic(z_low,  country)
    du_for_hi = delta_u_foreign(z_high, country)
    du_for_lo = delta_u_foreign(z_low,  country)

    p_dom_hi = sigmoid(du_dom_hi)
    p_dom_lo = sigmoid(du_dom_lo)
    p_for_hi = sigmoid(du_for_hi)
    p_for_lo = sigmoid(du_for_lo)

    L.append("    Domestic CO₂ framing:")
    L.append(fmt_p(p_dom_lo, f"P | z = {z_low}  (low)"))
    L.append(fmt_p(p_dom_hi, f"P | z = {z_high} (high)"))
    L.append(fmt_dp(p_dom_hi - p_dom_lo, "ΔP (high − low)"))
    blank()
    L.append("    Foreign CO₂ framing:")
    L.append(fmt_p(p_for_lo, f"P | z = {z_low}  (low)"))
    L.append(fmt_p(p_for_hi, f"P | z = {z_high} (high)"))
    L.append(fmt_dp(p_for_hi - p_for_lo, "ΔP (high − low)"))
    blank()

# ---- Approach C ----

h2("Approach C  |  z* sampled from conjugate posterior given item responses")
blank()
L.append("  Method: per posterior draw, z* is sampled from N(μ*, s*) where μ* and s*")
L.append("  are derived analytically from the specified Likert responses and that draw's")
L.append("  λ and σ values. This propagates measurement uncertainty through to P.")
blank()
L.append("  Measurement model:  y_k ~ N(λ_k × z, σ),  λ_1 = 1 by convention.")
L.append("  Posterior:  z* | y ~ N(μ*, s*),")
L.append("    μ* = (y₁ + λ₁y₂ + λ₂y₃) / (σ² + 1 + λ₁² + λ₂²)")
L.append("    s* = σ  / √(σ² + 1 + λ₁² + λ₂²)")
blank()
L.append(f"  y_high = {y_high.tolist()}  (all items at maximum, e.g. 1–5 scale)")
L.append(f"  y_low  = {y_low.tolist()}  (all items at minimum)")
blank()

mu_hi_draws, s_hi_draws = z_moments(y_high)
mu_lo_draws, s_lo_draws = z_moments(y_low)
L.append("  Implied z* posterior (across posterior draws):")
L.append(f"    y_high → E[z*] median: {float(np.median(mu_hi_draws)):+.3f}   s* median: {float(np.median(s_hi_draws)):.3f}")
L.append(f"    y_low  → E[z*] median: {float(np.median(mu_lo_draws)):+.3f}   s* median: {float(np.median(s_lo_draws)):.3f}")
blank()

np.random.seed(42)
for country in countries:
    L.append(f"  {country.capitalize()}")
    blank()

    z_hi_c = z_from_items(y_high)
    z_lo_c = z_from_items(y_low)

    p_dom_hi_c = sigmoid(delta_u_domestic(z_hi_c, country))
    p_dom_lo_c = sigmoid(delta_u_domestic(z_lo_c, country))
    p_for_hi_c = sigmoid(delta_u_foreign(z_hi_c, country))
    p_for_lo_c = sigmoid(delta_u_foreign(z_lo_c, country))

    L.append("    Domestic CO₂ framing:")
    L.append(fmt_p(p_dom_lo_c, "P | y_low"))
    L.append(fmt_p(p_dom_hi_c, "P | y_high"))
    L.append(fmt_dp(p_dom_hi_c - p_dom_lo_c, "ΔP (high − low)"))
    blank()
    L.append("    Foreign CO₂ framing:")
    L.append(fmt_p(p_for_lo_c, "P | y_low"))
    L.append(fmt_p(p_for_hi_c, "P | y_high"))
    L.append(fmt_dp(p_for_hi_c - p_for_lo_c, "ΔP (high − low)"))
    blank()

hr("=")

output = "\n".join(L)
print(output)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"\nSaved to {out_path}")
