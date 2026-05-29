import os
import arviz as az
import pandas as pd
import numpy as np

try:
    run_basic = snakemake.config.get("run_basic_model", True)
    bcm_path  = snakemake.input.bcm if run_basic else None
    hcm_path  = snakemake.input.hcm
    out_beta    = snakemake.output.beta
    out_country = snakemake.output.country
    out_theta   = snakemake.output.theta
except NameError:
    run_basic = True
    bcm_path  = "output/data/inference_basic_choice.nc"
    hcm_path  = "output/data/inference_hybrid_choice.nc"
    out_beta    = "output/data/posteriors_beta.csv"
    out_country = "output/data/posteriors_country.csv"
    out_theta   = "output/data/posteriors_theta.csv"


def extract_beta(posterior, model_name):
    """
    Partworth utilities per attribute level.

    For all levels: posterior samples of beta (average across framings and
    countries). For attr_source_purpose non-baseline (foreign) levels: also
    computes within-framing contrasts (foreign - domestic under the same
    framing) for source and purpose framing conditions. Using within-framing
    contrasts ensures the correct reference: foreign vs domestic when BOTH
    are framed identically.

    sum_to_zero: within-framing contrasts computed explicitly.
    reference_level: domestic = 0 by construction, so the foreign utility
    already is the within-framing contrast; no adjustment needed.

    Columns: model, level, framing, chain, draw, value
    """
    beta = posterior["beta"]   # (chain, draw, level)
    delta = posterior["delta"] # (chain, draw) — scalar, source_purpose only

    df = beta.to_dataframe(name="value").reset_index()[["chain", "draw", "level", "value"]]
    df["model"] = model_name
    df["framing"] = "average"

    sp_levels = [l for l in beta.level.values if "source_purpose" in str(l)]
    sp_contrast = (
        {str(l): (-0.5 if i == 0 else 0.5) for i, l in enumerate(sp_levels)}
        if len(sp_levels) > 1 else
        {str(l): 1.0 for l in sp_levels}
    )

    framing_rows = []
    n_chains  = len(beta.chain)
    n_draws   = len(beta.draw)
    chain_idx = np.repeat(beta.chain.values, n_draws)
    draw_idx  = np.tile(beta.draw.values, n_chains)

    if len(sp_levels) > 1:
        # sum_to_zero: within-framing contrast = foreign_framing - domestic_framing
        domestic_l        = sp_levels[0]
        domestic_contrast = sp_contrast[str(domestic_l)]  # -0.5
        beta_dom          = beta.sel(level=domestic_l)

        for level in sp_levels[1:]:  # non-baseline (foreign) levels only
            beta_sp  = beta.sel(level=level)
            contrast = sp_contrast[str(level)]
            for framing, f_val in [("purpose", -0.5), ("source", 0.5)]:
                within = (
                    (beta_sp  + delta * contrast          * f_val) -
                    (beta_dom + delta * domestic_contrast * f_val)
                ).values.ravel()
                framing_rows.append(pd.DataFrame({
                    "chain":   chain_idx,
                    "draw":    draw_idx,
                    "level":   f"{level}_{framing}",
                    "value":   within,
                    "model":   model_name,
                    "framing": framing,
                }))
    else:
        # reference_level: domestic = 0 always; foreign utility is already the contrast
        for level in sp_levels:
            beta_sp  = beta.sel(level=level)
            contrast = sp_contrast[str(level)]
            for framing, f_val in [("purpose", -0.5), ("source", 0.5)]:
                adjusted = (beta_sp + delta * contrast * f_val).values.ravel()
                framing_rows.append(pd.DataFrame({
                    "chain":   chain_idx,
                    "draw":    draw_idx,
                    "level":   f"{level}_{framing}",
                    "value":   adjusted,
                    "model":   model_name,
                    "framing": framing,
                }))

    if framing_rows:
        df = pd.concat([df] + framing_rows, ignore_index=True)

    return df[["model", "level", "framing", "chain", "draw", "value"]]


def extract_country(posterior, model_name):
    """
    Country-specific utilities: beta + gamma[country, level], with framing
    adjustment for attr_source_purpose levels.

    For all non-source_purpose levels: framing = "average".
    For source_purpose levels: framing-specific rows (source / purpose) replace
    the average, computed as beta + gamma + delta * f.

    Columns: model, country, level, framing, chain, draw, value
    """
    beta = posterior["beta"]   # (chain, draw, level)
    gamma = posterior["gamma"] # (chain, draw, country, level)
    delta = posterior["delta"] # (chain, draw) — scalar

    sp_levels   = [l for l in beta.level.values if "source_purpose" in str(l)]
    # fix: match extract_beta logic so reference_level gets contrast 1.0 not -0.5
    sp_contrast = (
        {str(l): (-0.5 if i == 0 else 0.5) for i, l in enumerate(sp_levels)}
        if len(sp_levels) > 1 else
        {str(l): 1.0 for l in sp_levels}
    )
    n_chains    = len(beta.chain)
    n_draws     = len(beta.draw)
    chain_idx   = np.repeat(beta.chain.values, n_draws)
    draw_idx    = np.tile(beta.draw.values, n_chains)

    rows = []
    for country in gamma.country.values:
        base = beta + gamma.sel(country=country)  # (chain, draw, level)

        # Average utility for all levels
        df = base.to_dataframe(name="value").reset_index()[["chain", "draw", "level", "value"]]
        df["model"]   = model_name
        df["country"] = str(country)
        df["framing"] = "average"
        rows.append(df)

        # Within-framing contrasts for source_purpose levels
        if len(sp_levels) > 1:
            # sum_to_zero: within-framing contrast = foreign_framing - domestic_framing
            domestic_l        = sp_levels[0]
            domestic_contrast = sp_contrast[str(domestic_l)]
            base_dom          = base.sel(level=domestic_l)

            for level in sp_levels[1:]:
                base_sp  = base.sel(level=level)
                contrast = sp_contrast[str(level)]
                for framing, f_val in [("purpose", -0.5), ("source", 0.5)]:
                    within = (
                        (base_sp  + delta * contrast          * f_val) -
                        (base_dom + delta * domestic_contrast * f_val)
                    ).values.ravel()
                    rows.append(pd.DataFrame({
                        "chain":   chain_idx,
                        "draw":    draw_idx,
                        "level":   f"{level}_{framing}",
                        "value":   within,
                        "model":   model_name,
                        "country": str(country),
                        "framing": framing,
                    }))
        else:
            # reference_level: domestic = 0; foreign utility is already the contrast
            for level in sp_levels:
                base_sp  = base.sel(level=level)
                contrast = sp_contrast[str(level)]
                for framing, f_val in [("purpose", -0.5), ("source", 0.5)]:
                    adjusted = (base_sp + delta * contrast * f_val).values.ravel()
                    rows.append(pd.DataFrame({
                        "chain":   chain_idx,
                        "draw":    draw_idx,
                        "level":   f"{level}_{framing}",
                        "value":   adjusted,
                        "model":   model_name,
                        "country": str(country),
                        "framing": framing,
                    }))

    result = pd.concat(rows, ignore_index=True)
    return result[["model", "country", "level", "framing", "chain", "draw", "value"]]


def extract_theta(posterior, model_name):
    """
    Value moderation effects: how much each latent value dimension shifts
    preference for each attribute level (theta_lreco, theta_galtan, theta_ecol).

    Columns: model, dim, level, chain, draw, value
    """
    rows = []
    for dim in ["lreco", "galtan", "ecol"]:
        param = f"theta_{dim}"
        if param not in posterior:
            continue
        df = posterior[param].to_dataframe(name="value").reset_index()[["chain", "draw", "level", "value"]]
        df["model"] = model_name
        df["dim"] = dim
        rows.append(df)

    result = pd.concat(rows, ignore_index=True)
    return result[["model", "dim", "level", "chain", "draw", "value"]]


# load inference data
hcm = az.from_netcdf(hcm_path)

if run_basic:
    bcm = az.from_netcdf(bcm_path)
    beta_all = pd.concat([
        extract_beta(bcm.posterior, "basic"),
        extract_beta(hcm.posterior, "hybrid"),
    ], ignore_index=True)
    country_all = pd.concat([
        extract_country(bcm.posterior, "basic"),
        extract_country(hcm.posterior, "hybrid"),
    ], ignore_index=True)
else:
    beta_all    = extract_beta(hcm.posterior, "hybrid")
    country_all = extract_country(hcm.posterior, "hybrid")

# theta only exists in the hybrid model
theta_all = extract_theta(hcm.posterior, "hybrid")

# save
os.makedirs(os.path.dirname(out_beta), exist_ok=True)
beta_all.to_csv(out_beta, index=False)
country_all.to_csv(out_country, index=False)
theta_all.to_csv(out_theta, index=False)
