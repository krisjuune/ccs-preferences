import os
import arviz as az
import pandas as pd
import numpy as np

try:
    run_basic = snakemake.config.get("run_basic_model", True)
    bcm_path  = snakemake.input.bcm if run_basic else None
    hcm_path  = snakemake.input.hcm
    hcm_input_path = snakemake.input.hcm_input
    out_beta    = snakemake.output.beta
    out_country = snakemake.output.country
    out_country_total = snakemake.output.country_total
    out_theta   = snakemake.output.theta
    out_loadings = snakemake.output.loadings
    out_alpha   = snakemake.output.alpha
    out_alpha_individual = snakemake.output.alpha_individual
except NameError:
    run_basic = True
    bcm_path  = "output/data/inference_basic_choice.nc"
    hcm_path  = "output/data/inference_main_hybrid_choice.nc"
    hcm_input_path = "data/hcm_input.csv"
    out_beta    = "output/data/posteriors_beta.csv"
    out_country = "output/data/posteriors_country.csv"
    out_country_total = "output/data/posteriors_country_total.csv"
    out_theta   = "output/data/posteriors_theta.csv"
    out_loadings = "output/data/posteriors_loadings.csv"
    out_alpha   = "output/data/posteriors_alpha.csv"
    out_alpha_individual = "output/data/posteriors_alpha_individual.csv"


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


def extract_indirect(posterior, df, model_name):
    """
    Indirect country effect mediated by the latent value dimensions: the
    portion of a country's average utility for a level that arises because
    that country has a systematically different distribution of latent
    values (lreco/galtan/ecol), as opposed to the direct/residual country
    effect already captured by gamma.

    indirect[country, level] = sum_dim theta_dim[level] *
        (mean of dim's latent trait within country - global mean across
         all individuals)

    Levels with no theta term (attribute baselines, dropped under
    reference_level coding) get indirect = 0, consistent with their direct
    effect also being 0 by construction.

    Columns: model, country, level, chain, draw, value
    """
    dim_latent = {
        "lreco": "lreco_latent",
        "galtan": "galtan_latent",
        "ecol": "socio_ecol_latent",
    }
    id_to_country = df.drop_duplicates("id").set_index("id")["country"]
    countries = sorted(id_to_country.unique())

    totals = {country: 0 for country in countries}
    level_coord = None

    for dim, latent_param in dim_latent.items():
        theta_param = f"theta_{dim}"
        if latent_param not in posterior or theta_param not in posterior:
            continue
        latent = posterior[latent_param]  # (chain, draw, individual)
        theta  = posterior[theta_param]   # (chain, draw, level)
        level_coord = theta.level

        global_mean = latent.mean(dim="individual")  # (chain, draw)

        for country in countries:
            ids_in_country = id_to_country[id_to_country == country].index.values
            country_mean = latent.sel(individual=ids_in_country).mean(dim="individual")
            centered = country_mean - global_mean       # (chain, draw)
            totals[country] = totals[country] + theta * centered  # (chain, draw, level)

    rows = []
    for country, contribution in totals.items():
        if level_coord is None:
            continue
        df_c = (
            contribution.to_dataframe(name="value")
            .reset_index()[["chain", "draw", "level", "value"]]
        )
        df_c["model"] = model_name
        df_c["country"] = country
        rows.append(df_c)

    result = pd.concat(rows, ignore_index=True)
    return result[["model", "country", "level", "chain", "draw", "value"]]


def combine_total(direct_df, indirect_df):
    """
    Total country effect = direct (beta + gamma, with framing adjustment for
    attr_source_purpose) + indirect (mediated by latent values). Indirect is
    matched on the level with any framing suffix stripped, since theta does
    not depend on framing.

    Columns: model, country, level, framing, chain, draw, value
    """
    indirect_df = indirect_df.rename(columns={"level": "base_level", "value": "indirect"})
    direct_df = direct_df.copy()
    direct_df["base_level"] = direct_df["level"].str.replace(r"_(source|purpose)$", "", regex=True)

    merged = direct_df.merge(
        indirect_df[["model", "country", "base_level", "chain", "draw", "indirect"]],
        on=["model", "country", "base_level", "chain", "draw"],
        how="left",
    )
    merged["indirect"] = merged["indirect"].fillna(0.0)
    merged["value"] = merged["value"] + merged["indirect"]
    return merged[["model", "country", "level", "framing", "chain", "draw", "value"]]


def extract_loadings(posterior, model_name):
    """
    Standardized factor loadings from the measurement model (SEM part of the HCM).

    Item 1 per dimension is fixed to λ=1 (unstandardized) for identification;
    its standardized value is 1 / sqrt(1 + σ²_likert), which varies per draw.
    Items 2 and 3: λ_std = λ / sqrt(λ² + σ²_likert).

    Columns: model, dim, item, chain, draw, value
    """
    rows = []
    dim_params = {"lreco": "lambda_lreco", "galtan": "lambda_galtan", "ecol": "lambda_ecol"}
    sigma    = posterior["likert_sigma"]      # (chain, draw)
    sigma_sq = sigma ** 2

    for dim, param in dim_params.items():
        if param not in posterior:
            continue
        lam      = posterior[param]  # (chain, draw, 2) — items 2 and 3
        item_dim = lam.dims[-1]

        # Item 1: fixed λ=1, standardized = 1 / sqrt(1 + σ²)
        lam_std_1 = 1.0 / np.sqrt(1.0 + sigma_sq)
        df1 = (
            lam_std_1.to_dataframe(name="value")
            .reset_index()[["chain", "draw", "value"]]
        )
        df1["model"] = model_name
        df1["dim"]   = dim
        df1["item"]  = "item_1"
        rows.append(df1)

        # Items 2 and 3: standardized = λ / sqrt(λ² + σ²)
        for i, item in enumerate(["item_2", "item_3"]):
            lam_i     = lam.isel({item_dim: i})
            lam_std_i = lam_i / np.sqrt(lam_i ** 2 + sigma_sq)
            df = (
                lam_std_i.to_dataframe(name="value")
                .reset_index()[["chain", "draw", "value"]]
            )
            df["model"] = model_name
            df["dim"]   = dim
            df["item"]  = item
            rows.append(df)

    result = pd.concat(rows, ignore_index=True)
    return result[["model", "dim", "item", "chain", "draw", "value"]]


def extract_alpha(posterior, model_name):
    """
    Left-choice positional bias (per-person, hierarchical; only present in
    main_hybrid_choice_model and full_interaction_choice_model).

    Population-level (alpha_mu, alpha_sigma): raw posterior samples, for a
    halfeye-style look at the overall left-choice tendency and how much it
    varies across people.

    Per-individual (alpha): posterior mean and SD only, not full draws —
    individual x draw would be far too large (~3000 individuals) for a
    quick-look CSV.

    Returns (population_df, individual_df).
    Population columns: model, param, chain, draw, value
    Individual columns: model, individual, mean, sd
    """
    pop_rows = []
    for param in ["alpha_mu", "alpha_sigma"]:
        if param not in posterior:
            continue
        df = posterior[param].to_dataframe(name="value").reset_index()[["chain", "draw", "value"]]
        df["model"] = model_name
        df["param"] = param
        pop_rows.append(df)
    population_df = (
        pd.concat(pop_rows, ignore_index=True)[["model", "param", "chain", "draw", "value"]]
        if pop_rows else pd.DataFrame(columns=["model", "param", "chain", "draw", "value"])
    )

    individual_df = pd.DataFrame(columns=["model", "individual", "mean", "sd"])
    if "alpha" in posterior:
        alpha = posterior["alpha"]
        individual_df = pd.DataFrame({
            "model":      model_name,
            "individual": alpha.individual.values,
            "mean":       alpha.mean(dim=["chain", "draw"]).values,
            "sd":         alpha.std(dim=["chain", "draw"]).values,
        })[["model", "individual", "mean", "sd"]]

    return population_df, individual_df


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

# theta and factor loadings only exist in the hybrid model
theta_all    = extract_theta(hcm.posterior, "hybrid")
loadings_all = extract_loadings(hcm.posterior, "hybrid")

# left-choice positional bias (only present in main_hybrid_choice_model /
# full_interaction_choice_model — absent params are silently skipped)
alpha_pop_all, alpha_individual_all = extract_alpha(hcm.posterior, "hybrid")

# total country effect = direct (beta + gamma) + indirect (mediated by values)
hcm_input_df       = pd.read_csv(hcm_input_path)
indirect_all       = extract_indirect(hcm.posterior, hcm_input_df, "hybrid")
country_direct_hcm = country_all[country_all["model"] == "hybrid"]
country_total_all  = combine_total(country_direct_hcm, indirect_all)

# save
os.makedirs(os.path.dirname(out_beta), exist_ok=True)
beta_all.to_csv(out_beta, index=False)
country_all.to_csv(out_country, index=False)
country_total_all.to_csv(out_country_total, index=False)
theta_all.to_csv(out_theta, index=False)
loadings_all.to_csv(out_loadings, index=False)
alpha_pop_all.to_csv(out_alpha, index=False)
alpha_individual_all.to_csv(out_alpha_individual, index=False)
