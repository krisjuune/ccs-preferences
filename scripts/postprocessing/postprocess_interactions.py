"""
Postprocessing for the main hybrid choice model's proximity x Source/Purpose
x country three-way interaction.

Extracts:
  1. posteriors_interact_coefs.csv — raw posterior samples of beta_interact
     and (per-country) gamma_interact
  2. posteriors_interact_conditional.csv — proximity utility under domestic
     vs. foreign CO2, pooled and per country (see plot_interact_conditional.R)
"""
import os
import arviz as az
import numpy as np
import pandas as pd

try:
    idata_path   = snakemake.input.idata
    out_coefs    = snakemake.output.coefs
    out_cond     = snakemake.output.conditional
except NameError:
    idata_path   = "output/data/inference_main_hybrid_choice.nc"
    out_coefs    = "output/data/posteriors_interact_coefs.csv"
    out_cond     = "output/data/posteriors_interact_conditional.csv"

os.makedirs("output/data", exist_ok=True)

# ---- load ----

idata     = az.from_netcdf(idata_path)
posterior = idata.posterior

beta          = posterior["beta"]           # (chain, draw, level)
gamma         = posterior["gamma"]          # (chain, draw, country, level)
beta_interact = posterior["beta_interact"]  # (chain, draw, prox_source_interact)
gamma_interact = posterior["gamma_interact"] # (chain, draw, country, prox_source_interact)
delta         = posterior["delta"]          # (chain, draw)

n_chains  = len(beta.chain)
n_draws   = len(beta.draw)
chain_idx = np.repeat(beta.chain.values, n_draws)
draw_idx  = np.tile(beta.draw.values, n_chains)

# ---- 1. raw interaction coefficients ----

beta_rows = (
    beta_interact
    .to_dataframe(name="value")
    .reset_index()[["chain", "draw", "prox_source_interact", "value"]]
    .rename(columns={"prox_source_interact": "interact"})
    .assign(country="pooled", param="beta_interact")
)

gamma_rows = (
    gamma_interact
    .to_dataframe(name="value")
    .reset_index()[["chain", "draw", "country", "prox_source_interact", "value"]]
    .rename(columns={"prox_source_interact": "interact"})
    .assign(param="gamma_interact")
)

coefs_df = pd.concat([beta_rows, gamma_rows], ignore_index=True)

# ---- 2. conditional marginal effects ----

interact_names = [str(l) for l in beta_interact.prox_source_interact.values]

rows = []


def ravel_posterior(da):
    """Flatten (chain, draw, ...) DataArray to a 1-D numpy array."""
    return da.values.ravel()


def add_row(rows, chain_idx, draw_idx, interaction, conditioning_level,
            country, values, framing=None):
    rows.append(pd.DataFrame({
        "chain":              chain_idx,
        "draw":               draw_idx,
        "interaction":        interaction,
        "conditioning_level": conditioning_level,
        "country":            country,
        "framing":            framing,
        "value":              values,
    }))


# helper: get the non-baseline level name for a given attribute prefix
def nonbaseline_level(prefix, beta_levels):
    return [l for l in beta_levels if str(l).startswith(prefix)]


# ---- Proximity utility: domestic vs. foreign CO2 ----
# Domestic utility per proximity level:  β[prox]                (+ γ[c, prox])
# Foreign utility per proximity level:   β[prox] + β[foreign] + β_interact[prox × foreign]
#                                         (+ γ[c, prox] + γ[c, foreign] + γ_interact[c, prox × foreign])
# Baseline proximity (abroad) has no β/γ term of its own (reference level),
# so its domestic utility is 0 by construction; its foreign utility is just
# the source/purpose main (+ country) effect, with no interaction term.

source_levels = nonbaseline_level("attr_source_purpose", beta.level.values)
prox_levels   = nonbaseline_level("attr_vicinity", beta.level.values)

if source_levels and prox_levels:
    source_l     = str(source_levels[0])  # "attr_source_purpose_foreign"
    beta_foreign = beta.sel(level=source_l)
    all_prox     = ["attr_vicinity_abroad"] + [str(l) for l in prox_levels]

    for prox_l in all_prox:
        is_baseline = prox_l == "attr_vicinity_abroad"
        beta_prox   = None if is_baseline else beta.sel(level=prox_l)
        interact_name = None
        beta_int      = None
        if not is_baseline:
            interact_name = next(
                n for n in interact_names
                if n.startswith(prox_l) and "source_purpose" in n
            )
            beta_int = beta_interact.sel(prox_source_interact=interact_name)

        for country_name in ["pooled"] + [str(c) for c in gamma.country.values]:
            pooled = country_name == "pooled"

            # domestic
            if is_baseline:
                dom_vals = np.zeros(n_chains * n_draws)
            elif pooled:
                dom_vals = beta_prox.values.ravel()
            else:
                gamma_prox = gamma.sel(country=country_name, level=prox_l)
                dom_vals = (beta_prox + gamma_prox).values.ravel()
            add_row(rows, chain_idx, draw_idx, "prox_source", prox_l,
                    country_name, dom_vals, framing="domestic")

            # foreign
            if is_baseline:
                if pooled:
                    for_vals = beta_foreign.values.ravel()
                else:
                    gamma_foreign = gamma.sel(country=country_name, level=source_l)
                    for_vals = (beta_foreign + gamma_foreign).values.ravel()
            elif pooled:
                for_vals = (beta_prox + beta_foreign + beta_int).values.ravel()
            else:
                gamma_prox    = gamma.sel(country=country_name, level=prox_l)
                gamma_foreign = gamma.sel(country=country_name, level=source_l)
                gamma_int     = gamma_interact.sel(
                    country=country_name, prox_source_interact=interact_name
                )
                for_vals = (
                    beta_prox + gamma_prox + beta_foreign + gamma_foreign +
                    beta_int + gamma_int
                ).values.ravel()
            add_row(rows, chain_idx, draw_idx, "prox_source", prox_l,
                    country_name, for_vals, framing="foreign")

cond_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ---- save ----

coefs_df.to_csv(out_coefs, index=False)
cond_df.to_csv(out_cond, index=False)
print(f"Saved interaction coefficients: {out_coefs}")
print(f"Saved conditional effects:      {out_cond}")
