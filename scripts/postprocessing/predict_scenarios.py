"""
Prediction via posterior predictive sampling (scenario-based approach).

For each declared scenario, uses pm.sample_posterior_predictive to propagate
the full posterior through the utility formula without any new MCMC. Each
scenario is a contrast between two "options" (dicts of non-baseline
attribute -> level; omitted attributes are at their reference level, which
contributes 0 to utility by construction under reference-level coding),
optionally varied across low/high values of one latent value dimension.

Scenarios map onto the paper's four hypotheses:
  H1 (design_{inclusive,neutral,extractive}_vs_{abroad,region}) — can an
      inclusive-design package (cost responsibility on the polluter, high
      engagement, an honest siting rationale) offset local-siting aversion,
      independent of individual values? Three bundles: inclusive (favorable
      on every manipulable attribute), extractive (unfavorable on every
      manipulable attribute — a designed worst case, not a claim about
      real-world prevalence), and neutral (all attributes but vicinity at
      their reference level, i.e. proximity alone with no other design
      intervention). "Locally sited" is a single pooled quantity: P(choose
      your municipality OR your region over the comparator), computed as
      the equal-weight average of the two per-level sigmoid probabilities
      (a card can only realize one vicinity level at a time, so this is a
      mixture over which local level is shown, not a joint attribute
      combination). Two comparators: abroad (the model's reference level)
      and another region (a non-baseline level, also domestic but not
      local to the respondent).
  H2 (proximity_ecol_vs_abroad / proximity_ecol_vs_region) — does
      proximity aversion narrow as ecological orientation increases? Same
      pooled "locally sited" quantity and two comparators as H1, varied
      across low/high ecological orientation instead of design bundle.
      (A vicinity x source_purpose x country crossed version — domestic vs.
      foreign CO2 — was tried and found null: the ecol-conditioning effect
      is essentially identical either way, and not a direct test of H2, so
      it was reverted rather than kept as a permanent fourth dimension.)
  H3 (costs_lreco / costs_galtan / costs_ecol / source_lreco /
      source_galtan / source_ecol / reason_costefficient_lreco /
      reason_costefficient_galtan / reason_costefficient_ecol /
      reason_closesource_lreco / reason_closesource_galtan /
      reason_closesource_ecol) — does each value dimension moderate its
      matching attribute domain (economic/cultural values <-> cost
      responsibility; all three <-> CO2 source/purpose; none <-> siting
      rationale)? costs_ecol is included as a specificity check: H3
      predicts ecol should NOT move cost responsibility (unlike
      lreco/galtan), so a near-null costs_ecol alongside a real source_ecol
      effect is itself evidence for the domain-matching claim, not just a
      completeness addition. reason_* scenarios test both non-baseline
      siting-rationale levels (vs. reference "sparsely-populated"): per the
      theta_reason posteriors, only galtan x "close to source" clearly
      excludes zero — cost-efficient is null-ish for all three dimensions,
      close-to-source is borderline for lreco/galtan and null for ecol — so
      these scenarios are expected to look mostly null throughout,
      consistent with reason not being value-structured. source_* scenarios
      report P(choose domestic CO2 over foreign) — domestic as option A,
      foreign as option B — so that "higher value -> higher P" points the
      same direction as
      every other H2/H3 panel.
  H4 is addressed separately by scripts/tables/country_gap_decomposition.py,
      not by posterior-predictive scenarios.

Positional bias (alpha) is dropped throughout — it is a survey-mechanics
effect (tendency to click the left option) that is not meaningful for
attribute-based predictions.

Standalone script, not wired into the Snakefile.
Run: python scripts/postprocessing/predict_scenarios.py
Output: output/predictions/prediction_scenarios.txt
        output/predictions/prediction_scenarios.nc
        output/predictions/prediction_scenarios.csv
"""
import os
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
from datetime import datetime

try:
    idata_path = snakemake.input[0]
    out_txt    = snakemake.output.txt
    out_nc     = snakemake.output.nc
    out_csv    = snakemake.output.csv
except NameError:
    idata_path = "output/data/inference_main_hybrid_choice.nc"
    out_txt    = "output/predictions/prediction_scenarios.txt"
    out_nc     = "output/predictions/prediction_scenarios.nc"
    out_csv    = "output/predictions/prediction_scenarios.csv"

os.makedirs(os.path.dirname(out_txt), exist_ok=True)

# ---- scenario declarations ----
# option_a / option_b: dict of attribute -> non-baseline level name (suffix
# only, e.g. "your municipality" for attr_vicinity). Omitted attributes are
# at their reference level (0 contribution). vary_dim in {"lreco", "galtan",
# "ecol", None}; None means a single fixed point at the population mean
# (z = 0), not a low/high contrast.
#
# A scenario's "components" is a list of {option_a, option_b, weight}
# dicts; its P is the weight-average of the per-component sigmoid(delta U).
# Most scenarios have a single component (weight 1.0). H1's "locally sited"
# scenarios pool two components (your municipality / your region, weight
# 0.5 each) to express "choose A over B, where A is whichever local level
# is shown."

ATTR_PREFIX = {
    "engagement":    "attr_engagement",
    "vicinity":      "attr_vicinity",
    "industry":      "attr_industry",
    "costs":         "attr_costs",
    "reason":        "attr_reason",
    "source_purpose": "attr_source_purpose",
}


def levels_of(option):
    """Map {short_attr: level} -> [full dummy-column names]."""
    return [f"{ATTR_PREFIX[attr]}_{level}" for attr, level in option.items()]


def single(option_a, option_b=None):
    return [{"option_a": option_a, "option_b": option_b or {}, "weight": 1.0}]


LOCAL_LEVELS = ["your municipality", "your region"]

# Reason is omitted (reference: sparsely-populated) for the inclusive bundle
# — a neutral/objective siting justification, paired with fair process.
# The extractive bundle pairs an explicitly self-serving justification
# ("cheaper to build here") with poor process, reinforcing the taxpayer-
# funded / foreign-CO2 / fossil-source story already in that bundle. The
# neutral bundle changes nothing but vicinity — no design intervention
# either way, isolating the pure proximity effect at the population mean.
INCLUSIVE_BUNDLE  = {"costs": "polluting industry", "engagement": "vote"}
EXTRACTIVE_BUNDLE = {"source_purpose": "foreign", "industry": "gas with CCS",
                      "reason": "cost-efficient"}
NEUTRAL_BUNDLE    = {}


def local_components(bundle_extra, option_b):
    """Equal-weight components pooling your municipality / your region as
    the locally-sited option, against a shared comparator option_b."""
    return [
        {"option_a": {"vicinity": lvl, **bundle_extra}, "option_b": option_b, "weight": 0.5}
        for lvl in LOCAL_LEVELS
    ]


SCENARIOS = [
    dict(
        name="proximity_ecol_vs_abroad", hypothesis="H2",
        components=local_components(NEUTRAL_BUNDLE, {}),
        vary_dim="ecol",
    ),
    dict(
        name="proximity_ecol_vs_region", hypothesis="H2",
        components=local_components(NEUTRAL_BUNDLE, {"vicinity": "another region"}),
        vary_dim="ecol",
    ),
    dict(
        name="design_inclusive_vs_abroad", hypothesis="H1",
        components=local_components(INCLUSIVE_BUNDLE, {}),
        vary_dim=None,
    ),
    dict(
        name="design_neutral_vs_abroad", hypothesis="H1",
        components=local_components(NEUTRAL_BUNDLE, {}),
        vary_dim=None,
    ),
    dict(
        name="design_extractive_vs_abroad", hypothesis="H1",
        components=local_components(EXTRACTIVE_BUNDLE, {}),
        vary_dim=None,
    ),
    dict(
        name="design_inclusive_vs_region", hypothesis="H1",
        components=local_components(INCLUSIVE_BUNDLE, {"vicinity": "another region"}),
        vary_dim=None,
    ),
    dict(
        name="design_neutral_vs_region", hypothesis="H1",
        components=local_components(NEUTRAL_BUNDLE, {"vicinity": "another region"}),
        vary_dim=None,
    ),
    dict(
        name="design_extractive_vs_region", hypothesis="H1",
        components=local_components(EXTRACTIVE_BUNDLE, {"vicinity": "another region"}),
        vary_dim=None,
    ),
    dict(
        name="costs_lreco", hypothesis="H3",
        components=single({"costs": "polluting industry"}),
        vary_dim="lreco",
    ),
    dict(
        name="costs_galtan", hypothesis="H3",
        components=single({"costs": "polluting industry"}),
        vary_dim="galtan",
    ),
    dict(
        name="costs_ecol", hypothesis="H3",
        components=single({"costs": "polluting industry"}),
        vary_dim="ecol",
    ),
    dict(
        name="source_lreco", hypothesis="H3",
        components=single({}, {"source_purpose": "foreign"}),
        vary_dim="lreco",
    ),
    dict(
        name="source_galtan", hypothesis="H3",
        components=single({}, {"source_purpose": "foreign"}),
        vary_dim="galtan",
    ),
    dict(
        name="source_ecol", hypothesis="H3",
        components=single({}, {"source_purpose": "foreign"}),
        vary_dim="ecol",
    ),
    dict(
        name="reason_costefficient_lreco", hypothesis="H3",
        components=single({"reason": "cost-efficient"}),
        vary_dim="lreco",
    ),
    dict(
        name="reason_costefficient_galtan", hypothesis="H3",
        components=single({"reason": "cost-efficient"}),
        vary_dim="galtan",
    ),
    dict(
        name="reason_costefficient_ecol", hypothesis="H3",
        components=single({"reason": "cost-efficient"}),
        vary_dim="ecol",
    ),
    dict(
        name="reason_closesource_lreco", hypothesis="H3",
        components=single({"reason": "close to source"}),
        vary_dim="lreco",
    ),
    dict(
        name="reason_closesource_galtan", hypothesis="H3",
        components=single({"reason": "close to source"}),
        vary_dim="galtan",
    ),
    dict(
        name="reason_closesource_ecol", hypothesis="H3",
        components=single({"reason": "close to source"}),
        vary_dim="ecol",
    ),
]

Z_LOW, Z_HIGH = -1.5, 1.5   # SD units from population mean, latent scale
FRAMING_F = 0.0             # "average" framing (no source/purpose split)

# ---- load posterior ----

print("Loading posterior...")
idata     = az.from_netcdf(idata_path)
posterior = idata.posterior

level_coords    = [str(l) for l in posterior["beta"].level.values]
country_coords  = [str(c) for c in posterior["gamma"].country.values]
interact_coords = [str(i) for i in posterior["beta_interact"].prox_source_interact.values]
level_index     = {l: i for i, l in enumerate(level_coords)}
interact_index  = {i: j for j, i in enumerate(interact_coords)}

# ---- build prediction model ----
# Variable names must match the inference model posterior exactly so that
# pm.sample_posterior_predictive can substitute posterior draws by name.
# The priors declared here are placeholders only — they are never sampled.

coords = {
    "level":                level_coords,
    "country":              country_coords,
    "prox_source_interact": interact_coords,
}

print("Building prediction model...")
with pm.Model(coords=coords) as pred_model:
    beta           = pm.Normal("beta",           mu=0, sigma=2,   dims="level")
    gamma          = pm.Normal("gamma",          mu=0, sigma=1,   dims=["country", "level"])
    delta          = pm.Normal("delta",          mu=0, sigma=1)
    beta_interact  = pm.Normal("beta_interact",  mu=0, sigma=0.5, dims="prox_source_interact")
    gamma_interact = pm.Normal("gamma_interact", mu=0, sigma=0.3, dims=["country", "prox_source_interact"])
    theta_lreco    = pm.Normal("theta_lreco",    mu=0, sigma=1,   dims="level")
    theta_galtan   = pm.Normal("theta_galtan",   mu=0, sigma=1,   dims="level")
    theta_ecol     = pm.Normal("theta_ecol",     mu=0, sigma=1,   dims="level")

    theta_by_dim = {"lreco": theta_lreco, "galtan": theta_galtan, "ecol": theta_ecol}

    def option_utility(levels, theta, c_idx, z_val, f_val):
        """Utility of one option relative to the all-baseline reference."""
        u = 0.0
        prox_level = source_level = None
        for lvl in levels:
            i = level_index[lvl]
            u = u + beta[i] + gamma[c_idx, i] + theta[i] * z_val
            if lvl.startswith("attr_vicinity"):
                prox_level = lvl
            if lvl.startswith("attr_source_purpose"):
                source_level = lvl
                u = u + delta * f_val
        if prox_level and source_level:
            j = interact_index[f"{prox_level}::{source_level}"]
            u = u + beta_interact[j] + gamma_interact[c_idx, j]
        return u

    det_names = []
    for sc in SCENARIOS:
        theta = theta_by_dim[sc["vary_dim"]] if sc["vary_dim"] else theta_ecol
        z_pairs = [("low", Z_LOW), ("high", Z_HIGH)] if sc["vary_dim"] else [("fixed", 0.0)]

        for c_idx, country in enumerate(country_coords):
            for z_label, z_val in z_pairs:
                p_total = 0.0
                for comp in sc["components"]:
                    delta_u = (
                        option_utility(levels_of(comp["option_a"]), theta, c_idx, z_val, FRAMING_F)
                        - option_utility(levels_of(comp["option_b"]), theta, c_idx, z_val, FRAMING_F)
                    )
                    p_total = p_total + comp["weight"] * pm.math.sigmoid(delta_u)
                det_name = f"P_{sc['name']}_{country}_{z_label}"
                pm.Deterministic(det_name, p_total)
                det_names.append(det_name)

# ---- posterior predictive sampling ----

print("Sampling posterior predictive...")
pred_tree = pm.sample_posterior_predictive(
    idata,
    model=pred_model,
    var_names=det_names,
    predictions=True,
    random_seed=42,
)

pred_tree.to_netcdf(out_nc)
print(f"Saved NetCDF: {out_nc}")

# ---- extract results ----

preds = pred_tree["predictions"]

dfs = []
for sc in SCENARIOS:
    z_labels = ("low", "high") if sc["vary_dim"] else ("fixed",)
    for country in country_coords:
        for z_label in z_labels:
            vals = preds[f"P_{sc['name']}_{country}_{z_label}"].values  # (chain, draw)
            n_chains, n_draws = vals.shape
            dfs.append(pd.DataFrame({
                "chain":      np.repeat(np.arange(n_chains), n_draws),
                "draw":       np.tile(np.arange(n_draws), n_chains),
                "scenario":   sc["name"],
                "hypothesis": sc["hypothesis"],
                "country":    country,
                "z_level":    z_label,
                "prob":       vals.ravel(),
            }))
pred_df = pd.concat(dfs, ignore_index=True)
pred_df.to_csv(out_csv, index=False)
print(f"Saved CSV: {out_csv}")

# ---- helpers ----

def hdi_1d(arr, prob=0.89):
    arr = np.sort(arr)
    n   = len(arr)
    k   = int(np.floor(prob * n))
    widths = arr[k:] - arr[:n - k]
    i = int(np.argmin(widths))
    return float(arr[i]), float(arr[i + k])


def fmt_p(draws, label):
    med    = float(np.median(draws))
    lo, hi = hdi_1d(draws)
    return f"    {label:<36s}  median: {med:.3f}  89% HDI: [{lo:.3f}, {hi:.3f}]"


def fmt_dp(draws, label):
    med    = float(np.median(draws))
    lo, hi = hdi_1d(draws)
    p_pos  = float(np.mean(draws > 0))
    return (
        f"    {label:<36s}  median: {med:+.3f}  89% HDI: [{lo:+.3f}, {hi:+.3f}]"
        f"  P(ΔP > 0): {p_pos:.3f}"
    )


# ---- format output ----

L = []

def hr(char="-"): L.append(char * 70)
def h1(text):     hr("="); L.append(text); hr("=")
def h2(text):     hr("-"); L.append(text); hr("-")
def blank():      L.append("")

h1("Prediction (posterior predictive): scenario probabilities by hypothesis")
L.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
L.append(f"Posterior : {idata_path}")
blank()
L.append("Method")
L.append("  pm.sample_posterior_predictive forwards each posterior draw through the")
L.append("  utility formula in a PyMC prediction model. No new MCMC sampling.")
L.append("  Positional bias (alpha) dropped — survey mechanics, not attribute-driven.")
L.append(f"  z_low = {Z_LOW}, z_high = {Z_HIGH} (SD units from population mean, latent scale),")
L.append("  used only for scenarios that vary a value dimension.")
blank()

for hyp in sorted({sc["hypothesis"] for sc in SCENARIOS}):
    h1(f"Hypothesis {hyp}")
    blank()
    for sc in [s for s in SCENARIOS if s["hypothesis"] == hyp]:
        h2(f"Scenario: {sc['name']}")
        if len(sc["components"]) == 1:
            comp = sc["components"][0]
            L.append(f"  Option A: {comp['option_a'] or '(all reference levels)'}")
            L.append(f"  Option B: {comp['option_b'] or '(all reference levels)'}")
        else:
            L.append("  Option A (pooled, equal weight):")
            for comp in sc["components"]:
                L.append(
                    f"    weight {comp['weight']:.2f}: "
                    f"A={comp['option_a']}  B={comp['option_b'] or '(all reference levels)'}"
                )
        if sc["vary_dim"]:
            L.append(f"  Varying  : {sc['vary_dim']} (low={Z_LOW}, high={Z_HIGH})")
        else:
            L.append("  Varying  : none (fixed at population mean)")
        blank()

        for country in country_coords:
            L.append(f"  {country.capitalize()}")
            sub = pred_df[(pred_df.scenario == sc["name"]) & (pred_df.country == country)]
            if sc["vary_dim"]:
                p_lo = sub[sub.z_level == "low"]["prob"].values
                p_hi = sub[sub.z_level == "high"]["prob"].values
                L.append(fmt_p(p_lo, f"P(choose A over B) | z = {Z_LOW}  (low)"))
                L.append(fmt_p(p_hi, f"P(choose A over B) | z = {Z_HIGH} (high)"))
                L.append(fmt_dp(p_hi - p_lo, "ΔP (high − low)"))
            else:
                p_fixed = sub[sub.z_level == "fixed"]["prob"].values
                L.append(fmt_p(p_fixed, "P(choose A over B)"))
            blank()

hr("=")

output = "\n".join(L)
print(output)
with open(out_txt, "w") as f:
    f.write(output + "\n")
print(f"Saved text summary: {out_txt}")
