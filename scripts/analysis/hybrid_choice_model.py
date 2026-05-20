import os
import pymc as pm
import pandas as pd
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

try:
    _mcmc   = snakemake.config["mcmc"]
    _seed   = _mcmc["seed"]
    _draws  = _mcmc["draws"]
    _tune   = _mcmc["tune"]
    _chains = _mcmc["chains"]
    _cores  = _mcmc["cores"]
    _coding = snakemake.config.get("coding", "sum_to_zero")
    _output = snakemake.output[0]
except NameError:
    _seed = 42
    _draws, _tune, _chains, _cores = 250, 250, 4, 4
    _coding = "sum_to_zero"
    _output = "output/data/inference_hybrid_choice.nc"

_model_name = os.path.splitext(os.path.basename(_output))[0]
_out_data = os.path.join("output", "data", _model_name)
_out_ql = os.path.join("output", "quick_look", _model_name)
os.makedirs("output/data", exist_ok=True)
os.makedirs("output/quick_look", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)

# %% pymc bug workaround

import pytensor
import pytensor.tensor as pt
pytensor.config.cxx = '/usr/bin/clang++'

# %% 

df = pd.read_csv("data/hcm_input.csv")

# %% define attributes and baselines

attributes = [
    "attr_engagement",
    "attr_vicinity",
    "attr_industry",
    "attr_costs",
    "attr_reason",
    "attr_source_purpose"
]

baseline_dict = {
    "attr_engagement": "inform",
    "attr_vicinity": "abroad",
    "attr_industry": "waste incineration",
    "attr_costs": "taxpayer",
    "attr_reason": "sparsely-populated",
    "attr_source_purpose": "domestic"
}

# %% define dummies

# reorder each attribute column by making it categorical with the baseline first
for attr in attributes:
    baseline = baseline_dict[attr]
    df[attr] = pd.Categorical(df[attr], categories=[baseline] + 
                              [level for level in df[attr].unique() if level != baseline], 
                              ordered=True)

if _coding == "sum_to_zero":
    # all levels included; baselines ordered first within each attribute
    dummies = pd.get_dummies(df[attributes], drop_first=False)
    ordered_columns = []
    for attr in attributes:
        baseline_col = f"{attr}_{baseline_dict[attr]}"
        attr_cols    = [col for col in dummies.columns if col.startswith(attr)]
        ordered_columns.append(baseline_col)
        ordered_columns.extend([col for col in attr_cols if col != baseline_col])
    dummies = dummies[ordered_columns].loc[:, lambda d: ~d.columns.duplicated()]
    level_names = dummies.columns.tolist()
    attr_slices = [
        [i for i, col in enumerate(level_names) if col.startswith(attr)]
        for attr in attributes
    ]
    # framing contrast: ±0.5 so framing sums to zero across source_purpose levels
    sp_baseline_col = f"attr_source_purpose_{baseline_dict['attr_source_purpose']}"
    sp_contrast = np.array([
        -0.5 if col == sp_baseline_col else
        (0.5 if col.startswith("attr_source_purpose") else 0.0)
        for col in dummies.columns
    ])
else:
    # reference_level: baseline dropped, non-baseline levels in attribute order
    dummies = pd.get_dummies(df[attributes], drop_first=True)
    ordered_columns = []
    for attr in attributes:
        ordered_columns.extend([col for col in dummies.columns if col.startswith(attr)])
    dummies = dummies[ordered_columns].loc[:, lambda d: ~d.columns.duplicated()]
    # framing mask: 1 for the remaining (non-baseline) source_purpose level, 0 elsewhere
    sp_contrast = np.array([
        1.0 if col.startswith("attr_source_purpose") else 0.0
        for col in dummies.columns
    ])

df["framing"] = df["framing"].astype("category")
df["country"] = df["country"].astype("category")

df_left = df[df.package == 1].reset_index(drop=True)
df_right = df[df.package == 2].reset_index(drop=True)

# create individual index that matches task dimension
unique_individuals = df["id"].unique()
id_to_index = {id_: i for i, id_ in enumerate(unique_individuals)}
df["individual_idx"] = df["id"].map(id_to_index)
individual_idx = df_left["id"].map(id_to_index).values

# one row per individual for the measurement model (Likert items are individual-level, not task-level)
df_survey = df.drop_duplicates("id").sort_values("individual_idx").reset_index(drop=True)

# add dimensions as coords
coords = {
    "level": dummies.columns.tolist(),
    "task": np.arange(df_left.shape[0]),
    "framing": df["framing"].cat.categories,
    "country": df["country"].cat.categories,
    "individual": unique_individuals
}

# %% check coords

print("Coords:")
for k, v in coords.items():
    print(f"{k}: {len(v)} items")

print("\nFirst 13 individual_idx:", individual_idx[:13])
print("Number of tasks:", len(individual_idx))
print("Number of unique individuals:", len(unique_individuals))

# %% define HCM

with pm.Model(coords=coords) as hcm_model:

    # latent trait priors per individual
    lreco_latent = pm.Normal("lreco_latent", mu=0, sigma=1, dims="individual")
    galtan_latent = pm.Normal("galtan_latent", mu=0, sigma=1, dims="individual")
    socio_ecol_latent = pm.Normal("socio_ecol_latent", mu=0, sigma=1, dims="individual")

    # factor loadings — first item per factor fixed to 1 for scale identification
    lambda_lreco = pm.Normal("lambda_lreco", mu=1, sigma=0.5, shape=2)
    lambda_galtan = pm.Normal("lambda_galtan", mu=1, sigma=0.5, shape=2)
    lambda_ecol = pm.Normal("lambda_ecol", mu=1, sigma=0.5, shape=2)

    # estimated residual noise in Likert scores
    likert_sigma = pm.HalfNormal("likert_sigma", sigma=0.3)

    # measurement model — one observation per individual
    survey_idx = df_survey["individual_idx"].values

    pm.Normal("lreco_1", mu=lreco_latent[survey_idx], sigma=likert_sigma,
              observed=df_survey["lreco_1"].values)
    pm.Normal("lreco_2", mu=lambda_lreco[0] * lreco_latent[survey_idx], sigma=likert_sigma,
              observed=df_survey["lreco_2"].values)
    pm.Normal("lreco_3", mu=lambda_lreco[1] * lreco_latent[survey_idx], sigma=likert_sigma,
              observed=df_survey["lreco_3"].values)

    pm.Normal("galtan_1", mu=galtan_latent[survey_idx], sigma=likert_sigma,
              observed=df_survey["galtan_1"].values)
    pm.Normal("galtan_2", mu=lambda_galtan[0] * galtan_latent[survey_idx], sigma=likert_sigma,
              observed=df_survey["galtan_2"].values)
    pm.Normal("galtan_3", mu=lambda_galtan[1] * galtan_latent[survey_idx], sigma=likert_sigma,
              observed=df_survey["galtan_3"].values)

    pm.Normal("socio_ecological_1", mu=socio_ecol_latent[survey_idx], sigma=likert_sigma,
              observed=df_survey["socio_ecological_1"].values)
    pm.Normal("socio_ecological_2", mu=lambda_ecol[0] * socio_ecol_latent[survey_idx], sigma=likert_sigma,
              observed=df_survey["socio_ecological_2"].values)
    pm.Normal("socio_ecological_3", mu=lambda_ecol[1] * socio_ecol_latent[survey_idx], sigma=likert_sigma,
              observed=df_survey["socio_ecological_3"].values)

    # framing centred to -0.5/+0.5 so beta = average partworth across both framings
    f = df_left["framing"].cat.codes.values.astype("float32") - 0.5
    pm.Data("f", f, dims="task")

    c = df_left["country"].cat.codes.values
    pm.Data("c", c, dims="task")

    # observed choices
    observed_choice_left = df_left["chosen"].values
    pm.Data("observed_choice_left", observed_choice_left, dims="task")

    # attribute dummies
    attribute_levels_left = pm.Data(
        "attribute_levels_left", 
        dummies[df.package == 1].values, 
        dims=["task", "level"]
    )

    attribute_levels_right = pm.Data(
        "attribute_levels_right", 
        dummies[df.package == 2].values, 
        dims=["task", "level"]
    )

    if _coding == "sum_to_zero":
        # value moderation: sum-to-zero per attribute (all levels included)
        theta_lreco_raw = pm.Normal("theta_lreco_raw", mu=0, sigma=1, dims="level")
        theta_lreco = pm.Deterministic(
            "theta_lreco",
            pt.concatenate([theta_lreco_raw[idx] - theta_lreco_raw[idx].mean() for idx in attr_slices]),
            dims="level",
        )
        theta_galtan_raw = pm.Normal("theta_galtan_raw", mu=0, sigma=1, dims="level")
        theta_galtan = pm.Deterministic(
            "theta_galtan",
            pt.concatenate([theta_galtan_raw[idx] - theta_galtan_raw[idx].mean() for idx in attr_slices]),
            dims="level",
        )
        theta_ecol_raw = pm.Normal("theta_ecol_raw", mu=0, sigma=1, dims="level")
        theta_ecol = pm.Deterministic(
            "theta_ecol",
            pt.concatenate([theta_ecol_raw[idx] - theta_ecol_raw[idx].mean() for idx in attr_slices]),
            dims="level",
        )
        # main effects: sum-to-zero per attribute
        beta_raw = pm.Normal("beta_raw", mu=0, sigma=2, dims="level")
        beta = pm.Deterministic(
            "beta",
            pt.concatenate([beta_raw[idx] - beta_raw[idx].mean() for idx in attr_slices]),
            dims="level",
        )
    else:
        # value moderation: unconstrained (baseline anchors identification)
        theta_lreco = pm.Normal("theta_lreco", mu=0, sigma=1, dims="level")
        theta_galtan = pm.Normal("theta_galtan", mu=0, sigma=1, dims="level")
        theta_ecol   = pm.Normal("theta_ecol",   mu=0, sigma=1, dims="level")
        # main effects: unconstrained relative to dropped baseline
        beta = pm.Normal("beta", mu=0, sigma=2, dims="level")

    # framing effect on attr_source_purpose only (scalar — one attribute is framed)
    delta = pm.Normal("delta", mu=0, sigma=1)

    gamma_raw = pm.Normal("gamma_raw", mu=0, sigma=1, dims=["country", "level"])
    if _coding == "sum_to_zero":
        # two zero-sum constraints: across countries per level, and across
        # levels per attribute per country (both needed when all levels included)
        gamma_cc = gamma_raw - gamma_raw.mean(axis=0)
        gamma = pm.Deterministic(
            "gamma",
            pt.concatenate(
                [gamma_cc[:, idx] - gamma_cc[:, idx].mean(axis=1, keepdims=True)
                 for idx in attr_slices],
                axis=1,
            ),
            dims=["country", "level"],
        )
    else:
        # reference_level: only across-country zero-sum needed (baseline anchors levels)
        gamma = pm.Deterministic(
            "gamma",
            gamma_raw - gamma_raw.mean(axis=0),
            dims=["country", "level"],
        )

    beta_modulated = (
        beta
        + delta * sp_contrast * f[:, None]
        + gamma[c, :]
        + theta_lreco * (lreco_latent - lreco_latent.mean())[individual_idx][:, None]
        + theta_galtan * (galtan_latent - galtan_latent.mean())[individual_idx][:, None]
        + theta_ecol * (socio_ecol_latent - socio_ecol_latent.mean())[individual_idx][:, None]
    )

    # get utilities
    utility_left = pm.Deterministic("utility_left", 
        pm.math.sum(attribute_levels_left * beta_modulated, axis=1), dims = "task")

    utility_right = pm.Deterministic("utility_right", 
        pm.math.sum(attribute_levels_right * beta_modulated, axis=1), dims = "task")

    # logit probability
    prob_choice_left = pm.Deterministic("probability_choice_left", 
        pm.math.exp(utility_left) / (pm.math.exp(utility_left) + pm.math.exp(utility_right)),
        dims="task"
    )

    # likelihood
    pm.Bernoulli("choice_distribution", p=prob_choice_left, observed=observed_choice_left)

# %% get priors

priors = pm.sample_prior_predictive(
    draws=1000,
    model=hcm_model,
    random_seed=_seed,
)

# %% check priors

az.summary(
    priors,
    group="prior",
    var_names=["beta", "delta", "gamma", "theta_lreco", "theta_galtan", "theta_ecol"],
).to_csv(f"{_out_data}_prior_summary.csv")

# %% run model

# run model with MCMC with 1000 draws, 500 tune samples, and 4 chains on 6 cores
inference_data = pm.sample(
    model=hcm_model,
    draws=_draws,
    tune=_tune,
    chains=_chains,
    cores=_cores,
    random_seed=_seed,
    return_inferencedata=True,
    target_accept=0.9,
)

# %% diagnostics

_hcm_vars = ["beta", "delta", "gamma", "theta_lreco", "theta_galtan", "theta_ecol"]

az.summary(inference_data, var_names=_hcm_vars).to_csv(f"{_out_data}_summary.csv")

for var in _hcm_vars:
    az.plot_trace(inference_data, var_names=[var])
    plt.savefig(f"{_out_ql}_trace_{var}.png", bbox_inches="tight")
    plt.close("all")

for var in _hcm_vars:
    az.plot_forest(inference_data, var_names=[var], combined=True)
    plt.savefig(f"{_out_ql}_forest_{var}.png", bbox_inches="tight")
    plt.close("all")

# %% save to netcdf

inference_data.to_netcdf(_output)

# %%
