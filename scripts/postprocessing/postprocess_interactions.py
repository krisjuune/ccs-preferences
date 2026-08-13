"""
Postprocessing for the main hybrid choice model's proximity x Source/Purpose
x country three-way interaction.

Extracts:
  1. posteriors_interact_coefs.csv — raw posterior samples of beta_interact
     and (per-country) gamma_interact
  2. posteriors_interact_conditional.csv — proximity utility under domestic
     vs. foreign CO2, pooled and per country. Includes direct effects
     (beta + gamma + interaction) plus the indirect value-moderation component
     (theta_dim * country_mean_latent deviation), matching the total-effect
     computation in postprocessing.combine_total().
"""
import os
import arviz as az
import numpy as np
import pandas as pd

try:
    idata_path     = snakemake.input.idata
    hcm_input_path = snakemake.input.hcm_input
    out_coefs      = snakemake.output.coefs
    out_cond       = snakemake.output.conditional
except NameError:
    idata_path     = "output/data/inference_main_hybrid_choice.nc"
    hcm_input_path = "data/hcm_input.csv"
    out_coefs      = "output/data/posteriors_interact_coefs.csv"
    out_cond       = "output/data/posteriors_interact_conditional.csv"

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

# ---- indirect value-moderation component ----
# Mirrors postprocessing.extract_indirect(): for each country, the indirect
# effect for level l is sum_dim theta_dim[l] * (country_mean_latent - global_mean).
# The baseline proximity level (abroad) is not in the posterior under
# reference_level coding, so its indirect contribution is 0 by construction.

def compute_indirect(posterior, hcm_df, levels, n_chains, n_draws):
    """
    Returns {country: {level: np.ndarray (n_chains*n_draws,)}}
    """
    dim_latent = {
        "lreco":  "lreco_latent",
        "galtan": "galtan_latent",
        "ecol":   "socio_ecol_latent",
    }
    id_to_country = hcm_df.drop_duplicates("id").set_index("id")["country"]
    countries = sorted(id_to_country.unique())
    theta_levels = None
    totals = {c: 0 for c in countries}

    for dim, latent_param in dim_latent.items():
        theta_param = f"theta_{dim}"
        if latent_param not in posterior or theta_param not in posterior:
            continue
        latent = posterior[latent_param]   # (chain, draw, individual)
        theta  = posterior[theta_param]    # (chain, draw, level)
        theta_levels = theta.level.values
        global_mean = latent.mean(dim="individual")
        for country in countries:
            ids = id_to_country[id_to_country == country].index.values
            country_mean = latent.sel(individual=ids).mean(dim="individual")
            totals[country] = totals[country] + theta * (country_mean - global_mean)

    result = {}
    zeros = np.zeros(n_chains * n_draws)
    for country, contribution in totals.items():
        result[country] = {}
        for lvl in levels:
            if isinstance(contribution, int) or theta_levels is None:
                result[country][lvl] = zeros
            elif lvl in theta_levels:
                result[country][lvl] = contribution.sel(level=lvl).values.ravel()
            else:
                result[country][lvl] = zeros
    return result


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
#                                         + indirect[prox]
# Foreign utility per proximity level:   β[prox] + β[foreign] + β_interact[prox × foreign]
#                                         (+ γ[c, prox] + γ[c, foreign] + γ_interact[c, prox × foreign])
#                                         + indirect[prox] + indirect[foreign]
# Baseline proximity (abroad) has no β/γ/indirect term of its own (reference
# level), so its domestic utility is 0; its foreign utility is just the
# source/purpose main (+ country + indirect[foreign]) effect.

source_levels = nonbaseline_level("attr_source_purpose", beta.level.values)
prox_levels   = nonbaseline_level("attr_vicinity", beta.level.values)

if source_levels and prox_levels:
    source_l     = str(source_levels[0])  # "attr_source_purpose_foreign"
    beta_foreign = beta.sel(level=source_l)
    all_prox     = ["attr_vicinity_abroad"] + [str(l) for l in prox_levels]

    # Compute indirect effects for proximity levels + source_purpose_foreign
    hcm_df   = pd.read_csv(hcm_input_path)
    indirect = compute_indirect(
        posterior, hcm_df,
        levels=all_prox + [source_l],
        n_chains=n_chains, n_draws=n_draws,
    )

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
            ind_prox   = indirect.get(country_name, {}).get(prox_l,   np.zeros(n_chains * n_draws)) if not pooled else np.zeros(n_chains * n_draws)
            ind_source = indirect.get(country_name, {}).get(source_l, np.zeros(n_chains * n_draws)) if not pooled else np.zeros(n_chains * n_draws)

            # domestic
            if is_baseline:
                dom_vals = np.zeros(n_chains * n_draws)
            elif pooled:
                dom_vals = beta_prox.values.ravel()
            else:
                gamma_prox = gamma.sel(country=country_name, level=prox_l)
                dom_vals = (beta_prox + gamma_prox).values.ravel() + ind_prox
            add_row(rows, chain_idx, draw_idx, "prox_source", prox_l,
                    country_name, dom_vals, framing="domestic")

            # foreign
            if is_baseline:
                if pooled:
                    for_vals = beta_foreign.values.ravel()
                else:
                    gamma_foreign = gamma.sel(country=country_name, level=source_l)
                    for_vals = (beta_foreign + gamma_foreign).values.ravel() + ind_source
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
                ).values.ravel() + ind_prox + ind_source
            add_row(rows, chain_idx, draw_idx, "prox_source", prox_l,
                    country_name, for_vals, framing="foreign")

cond_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ---- save ----

coefs_df.to_csv(out_coefs, index=False)
cond_df.to_csv(out_cond, index=False)
print(f"Saved interaction coefficients: {out_coefs}")
print(f"Saved conditional effects:      {out_cond}")
