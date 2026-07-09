"""
Main hybrid choice model: the HCM (measurement model + value-moderated
choice model) plus the proximity x Source/Purpose x country three-way
interaction. This is the model used for all of the repo's main results and
plots (partworths, country direct/total effects, theta, factor loadings),
since omitting a real interaction can bias the main-effect estimates.

Also includes a per-person, hierarchical left-choice positional bias (alpha):
the tendency to pick the left-hand alternative regardless of its attributes,
added only to utility_left (right is the reference).

Always uses reference level coding regardless of the global `coding` config
setting, because reference level gives clean, unambiguous interaction
estimates.

See base_hybrid_choice_model.py for the same model without the interaction
term (optional, for comparison; toggle via run_base_model in config.yaml).
"""
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

pytensor.config.cxx = "/usr/bin/clang++"

# ---- config ----

try:
    _mcmc   = snakemake.config["mcmc"]
    _seed   = _mcmc["seed"]
    _draws  = _mcmc["draws"]
    _tune   = _mcmc["tune"]
    _chains = _mcmc["chains"]
    _cores  = _mcmc["cores"]
    _output = snakemake.output[0]
except NameError:
    _seed = 42
    _draws, _tune, _chains, _cores = 250, 250, 4, 4
    _output = "output/data/inference_main_hybrid_choice.nc"

_model_name = os.path.splitext(os.path.basename(_output))[0]
_out_data = os.path.join("output", "data", _model_name)
_out_ql   = os.path.join("output", "quick_look", _model_name)
os.makedirs("output/data", exist_ok=True)
os.makedirs("output/quick_look", exist_ok=True)

# ---- data ----

df = pd.read_csv("data/hcm_input.csv")

attributes = [
    "attr_engagement", "attr_vicinity", "attr_industry",
    "attr_costs", "attr_reason", "attr_source_purpose",
]
baseline_dict = {
    "attr_engagement":    "inform",
    "attr_vicinity":      "abroad",
    "attr_industry":      "waste incineration",
    "attr_costs":         "taxpayer",
    "attr_reason":        "sparsely-populated",
    "attr_source_purpose": "domestic",
}

# reference level coding (always for the main model)
for attr in attributes:
    baseline = baseline_dict[attr]
    df[attr] = pd.Categorical(
        df[attr],
        categories=[baseline] + [l for l in df[attr].unique() if l != baseline],
        ordered=True,
    )

dummies = pd.get_dummies(df[attributes], drop_first=True)
ordered_columns = []
for attr in attributes:
    ordered_columns.extend([c for c in dummies.columns if c.startswith(attr)])
dummies = dummies[ordered_columns].loc[:, ~dummies.columns.duplicated()]

# framing mask: 1 for source_purpose non-baseline (foreign), 0 elsewhere
sp_mask = np.array(
    [1.0 if c.startswith("attr_source_purpose") else 0.0 for c in dummies.columns]
)

# ---- proximity x source/purpose interaction dummies (basis for the three-way) ----

prox_cols   = [c for c in dummies.columns if c.startswith("attr_vicinity")]
source_cols = [c for c in dummies.columns if c.startswith("attr_source_purpose")]

prox_source_cols = []
interact_parts = {}
for pc in prox_cols:
    for sc in source_cols:
        name = f"{pc}::{sc}"
        interact_parts[name] = dummies[pc] * dummies[sc]
        prox_source_cols.append(name)

interact_df = pd.DataFrame(interact_parts)

# ---- HCM setup ----

df["framing"] = df["framing"].astype("category")
df["country"] = df["country"].astype("category")
df_left = df[df.package == 1].reset_index(drop=True)

unique_individuals = df["id"].unique()
id_to_index = {id_: i for i, id_ in enumerate(unique_individuals)}
df["individual_idx"] = df["id"].map(id_to_index)
individual_idx = df_left["id"].map(id_to_index).values

df_survey = (
    df.drop_duplicates("id").sort_values("individual_idx").reset_index(drop=True)
)

coords = {
    "level":               ordered_columns,
    "task":                np.arange(df_left.shape[0]),
    "country":             df["country"].cat.categories,
    "individual":          unique_individuals,
    "prox_source_interact": prox_source_cols,
}

# ---- model ----

with pm.Model(coords=coords) as model:

    # -- measurement model (same as base HCM) --
    lreco_latent      = pm.Normal("lreco_latent",      mu=0, sigma=1, dims="individual")
    galtan_latent     = pm.Normal("galtan_latent",     mu=0, sigma=1, dims="individual")
    socio_ecol_latent = pm.Normal("socio_ecol_latent", mu=0, sigma=1, dims="individual")

    lambda_lreco  = pm.Normal("lambda_lreco",  mu=1, sigma=0.5, shape=2)
    lambda_galtan = pm.Normal("lambda_galtan", mu=1, sigma=0.5, shape=2)
    lambda_ecol   = pm.Normal("lambda_ecol",   mu=1, sigma=0.5, shape=2)
    likert_sigma  = pm.HalfNormal("likert_sigma", sigma=0.3)

    s = df_survey["individual_idx"].values
    pm.Normal("lreco_1", mu=lreco_latent[s], sigma=likert_sigma, observed=df_survey["lreco_1"].values)
    pm.Normal("lreco_2", mu=lambda_lreco[0] * lreco_latent[s], sigma=likert_sigma, observed=df_survey["lreco_2"].values)
    pm.Normal("lreco_3", mu=lambda_lreco[1] * lreco_latent[s], sigma=likert_sigma, observed=df_survey["lreco_3"].values)
    pm.Normal("galtan_1", mu=galtan_latent[s], sigma=likert_sigma, observed=df_survey["galtan_1"].values)
    pm.Normal("galtan_2", mu=lambda_galtan[0] * galtan_latent[s], sigma=likert_sigma, observed=df_survey["galtan_2"].values)
    pm.Normal("galtan_3", mu=lambda_galtan[1] * galtan_latent[s], sigma=likert_sigma, observed=df_survey["galtan_3"].values)
    pm.Normal("socio_ecological_1", mu=socio_ecol_latent[s], sigma=likert_sigma, observed=df_survey["socio_ecological_1"].values)
    pm.Normal("socio_ecological_2", mu=lambda_ecol[0] * socio_ecol_latent[s], sigma=likert_sigma, observed=df_survey["socio_ecological_2"].values)
    pm.Normal("socio_ecological_3", mu=lambda_ecol[1] * socio_ecol_latent[s], sigma=likert_sigma, observed=df_survey["socio_ecological_3"].values)

    # -- framing and country --
    f = pm.Data("f", df_left["framing"].cat.codes.values.astype("float32") - 0.5, dims="task")
    c = pm.Data("c", df_left["country"].cat.codes.values, dims="task")
    pm.Data("observed_choice_left", df_left["chosen"].values, dims="task")

    attribute_levels_left  = pm.Data("attribute_levels_left",  dummies[df.package == 1].values, dims=["task", "level"])
    attribute_levels_right = pm.Data("attribute_levels_right", dummies[df.package == 2].values, dims=["task", "level"])

    # prox x source interaction levels (used for both beta_interact and the
    # country-moderated three-way via gamma_interact)
    prox_source_left  = pm.Data("prox_source_left",  interact_df[df.package == 1].values, dims=["task", "prox_source_interact"])
    prox_source_right = pm.Data("prox_source_right", interact_df[df.package == 2].values, dims=["task", "prox_source_interact"])

    # -- value moderation (unconstrained: reference level anchors identification) --
    theta_lreco  = pm.Normal("theta_lreco",  mu=0, sigma=1, dims="level")
    theta_galtan = pm.Normal("theta_galtan", mu=0, sigma=1, dims="level")
    theta_ecol   = pm.Normal("theta_ecol",   mu=0, sigma=1, dims="level")

    # -- main effects (reference level) --
    beta  = pm.Normal("beta",  mu=0, sigma=2, dims="level")
    delta = pm.Normal("delta", mu=0, sigma=1)

    # -- country main effects (zero-sum across countries) --
    gamma_raw = pm.Normal("gamma_raw", mu=0, sigma=1, dims=["country", "level"])
    gamma = pm.Deterministic(
        "gamma", gamma_raw - gamma_raw.mean(axis=0), dims=["country", "level"]
    )

    # -- proximity x source/purpose interaction (pooled) --
    beta_interact = pm.Normal("beta_interact", mu=0, sigma=0.5, dims="prox_source_interact")

    # -- three-way: country x proximity x source (zero-sum across countries) --
    gamma_interact_raw = pm.Normal(
        "gamma_interact_raw", mu=0, sigma=0.3, dims=["country", "prox_source_interact"]
    )
    gamma_interact = pm.Deterministic(
        "gamma_interact",
        gamma_interact_raw - gamma_interact_raw.mean(axis=0),
        dims=["country", "prox_source_interact"],
    )

    # -- left-choice positional bias (per-person, hierarchical) --
    # captures the tendency to pick the left-hand alternative regardless of
    # its attributes; right is the reference (no constant added there)
    alpha_mu    = pm.Normal("alpha_mu", mu=0, sigma=1)
    alpha_sigma = pm.HalfNormal("alpha_sigma", sigma=0.5)
    alpha_raw   = pm.Normal("alpha_raw", mu=0, sigma=1, dims="individual")
    alpha = pm.Deterministic(
        "alpha", alpha_mu + alpha_sigma * alpha_raw, dims="individual"
    )

    # -- modulated beta (main effects + framing + country + values) --
    beta_mod = (
        beta
        + delta * sp_mask * f[:, None]
        + gamma[c, :]
        + theta_lreco  * (lreco_latent      - lreco_latent.mean())[individual_idx][:, None]
        + theta_galtan * (galtan_latent      - galtan_latent.mean())[individual_idx][:, None]
        + theta_ecol   * (socio_ecol_latent  - socio_ecol_latent.mean())[individual_idx][:, None]
    )

    # -- utilities --
    utility_left = pm.Deterministic("utility_left", (
        pm.math.sum(attribute_levels_left * beta_mod, axis=1)
        + pm.math.sum(prox_source_left * beta_interact, axis=1)
        + pm.math.sum(prox_source_left * gamma_interact[c, :], axis=1)
        + alpha[individual_idx]
    ), dims="task")

    utility_right = pm.Deterministic("utility_right", (
        pm.math.sum(attribute_levels_right * beta_mod, axis=1)
        + pm.math.sum(prox_source_right * beta_interact, axis=1)
        + pm.math.sum(prox_source_right * gamma_interact[c, :], axis=1)
    ), dims="task")

    prob_left = pm.Deterministic(
        "probability_choice_left",
        pm.math.exp(utility_left) / (pm.math.exp(utility_left) + pm.math.exp(utility_right)),
        dims="task",
    )
    pm.Bernoulli("choice_distribution", p=prob_left, observed=df_left["chosen"].values)

# ---- prior check ----

priors = pm.sample_prior_predictive(draws=1000, model=model, random_seed=_seed)
az.summary(priors, group="prior",
           var_names=["beta", "delta", "gamma", "beta_interact", "gamma_interact",
                      "alpha_mu", "alpha_sigma"]
           ).to_csv(f"{_out_data}_prior_summary.csv")

# ---- sample ----

inference_data = pm.sample(
    model=model, draws=_draws, tune=_tune, chains=_chains, cores=_cores,
    random_seed=_seed, return_inferencedata=True, target_accept=0.9,
)

# ---- diagnostics ----

_main_vars     = ["beta", "delta", "gamma", "theta_lreco", "theta_galtan", "theta_ecol",
                  "alpha_mu", "alpha_sigma"]
_interact_vars = ["beta_interact", "gamma_interact"]
_all_vars = _main_vars + _interact_vars

az.summary(inference_data, var_names=_all_vars).to_csv(f"{_out_data}_summary.csv")

for var in _all_vars:
    az.plot_trace(inference_data, var_names=[var])
    plt.savefig(f"{_out_ql}_trace_{var}.png", bbox_inches="tight")
    plt.close("all")

az.plot_forest(inference_data, var_names=_interact_vars, combined=True)
plt.savefig(f"{_out_ql}_forest_interactions.png", bbox_inches="tight")
plt.close("all")

inference_data.to_netcdf(_output)
