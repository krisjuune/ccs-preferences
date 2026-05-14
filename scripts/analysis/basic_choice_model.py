import os
import pymc as pm
import pandas as pd
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

try:
    _mcmc = snakemake.config["mcmc"]
    _draws = _mcmc["draws"]
    _tune = _mcmc["tune"]
    _chains = _mcmc["chains"]
    _cores = _mcmc["cores"]
    _output = snakemake.output[0]
except NameError:
    _draws, _tune, _chains, _cores = 250, 250, 4, 4
    _output = "output/data/inference_basic_choice.nc"

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

# %% define lists and translations

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

# generate dummies with all levels (no baseline dropped)
dummies = pd.get_dummies(df[attributes], drop_first=False)

# reorder columns: baseline first, then remaining levels, for each attribute
ordered_columns = []
for attr in attributes:
    baseline_col = f"{attr}_{baseline_dict[attr]}"
    attr_columns  = [col for col in dummies.columns if col.startswith(attr)]
    ordered_columns.append(baseline_col)
    ordered_columns.extend([col for col in attr_columns if col != baseline_col])

dummies = dummies[ordered_columns]
dummies = dummies.loc[:, ~dummies.columns.duplicated()]

# per-attribute index groups for sum-to-zero constraint on beta
level_names = dummies.columns.tolist()
attr_slices = [
    [i for i, col in enumerate(level_names) if col.startswith(attr)]
    for attr in attributes
]

# framing contrast for attr_source_purpose: -0.5 for baseline (domestic), +0.5 for others
# sums to zero across levels, consistent with the sum-to-zero constraint on beta
sp_baseline_col = f"attr_source_purpose_{baseline_dict['attr_source_purpose']}"
sp_contrast = np.array([
    -0.5 if col == sp_baseline_col else
    (0.5 if col.startswith("attr_source_purpose") else 0.0)
    for col in dummies.columns
])

df["framing"] = df["framing"].astype("category")
df["country"] = df["country"].astype("category")

coords = {
    "level": dummies.columns.values,
    "task": np.arange(len(df) // 2),
    "framing": df["framing"].cat.categories,
    "country": df["country"].cat.categories
}


# %% define model

with pm.Model(coords=coords) as bayes_model:
    
    # main effects — sum-to-zero within each attribute so beta = deviation from
    # attribute mean rather than deviation from an arbitrary baseline
    beta_raw = pm.Normal("beta_raw", mu=0, sigma=2, dims="level")
    beta = pm.Deterministic(
        "beta",
        pt.concatenate([beta_raw[idx] - beta_raw[idx].mean() for idx in attr_slices]),
        dims="level",
    )

    # framing effect on attr_source_purpose only (scalar — one attribute is framed)
    delta = pm.Normal("delta", mu=0, sigma=1)

    # country effect — two zero-sum constraints:
    # 1. across countries per level (so beta = true average across countries)
    # 2. across levels within each attribute per country (needed because all
    #    levels are included and adding a constant per attribute cancels in U_left - U_right)
    gamma_raw = pm.Normal("gamma_raw", mu=0, sigma=1, dims=["country", "level"])
    gamma_cc  = gamma_raw - gamma_raw.mean(axis=0)
    gamma = pm.Deterministic(
        "gamma",
        pt.concatenate(
            [gamma_cc[:, idx] - gamma_cc[:, idx].mean(axis=1, keepdims=True)
             for idx in attr_slices],
            axis=1,
        ),
        dims=["country", "level"],
    )
    
    # framing centred to -0.5/+0.5 so beta = average partworth across both framings
    f = pm.Data("f", df.loc[df.package == 1, "framing"].cat.codes.values.astype("float32") - 0.5, dims="task")
    country_idx = df.loc[df.package == 1, "country"].cat.codes.values
    c = pm.Data("c", country_idx, dims = "task")

    # observed choices
    observed_choice_left = pm.Data(
        "observed_choice_left", 
        df.loc[df.package == 1, "chosen"].values, 
        dims="task"
    )

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

    # compute modified coefficients depending on framing
    # this gives beta + delta * framing per task and level
    # adding country effect
    beta_framed = beta + delta * sp_contrast * f[:, None] + gamma[c, :]

    # compute utility
    utility_left = pm.Deterministic(
        "utility_left",
        pm.math.sum(attribute_levels_left * beta_framed, axis=1),
        dims="task"
    )

    utility_right = pm.Deterministic(
        "utility_right",
        pm.math.sum(attribute_levels_right * beta_framed, axis=1),
        dims="task"
    )

    # choice probability via logit
    probability_choice_left = pm.Deterministic(
        "probability_choice_left",
        pm.math.exp(utility_left) / (pm.math.exp(utility_left) + pm.math.exp(utility_right)),
        dims="task"
    )

    # likelihood
    choice_distribution = pm.Bernoulli(
        "choice_distribution", 
        p=probability_choice_left, 
        observed=observed_choice_left
    )

# %% get priors

priors = pm.sample_prior_predictive(
    draws=1000,
    model=bayes_model,
    random_seed=42,
)

# %% check priors

az.summary(priors, group="prior", var_names=["beta", "delta", "gamma"]).to_csv(
    f"{_out_data}_prior_summary.csv"
)

# run model with MCMC with 1000 draws, 500 tune samples, and 4 chains on 6 cores
inference_data = pm.sample(
    model=bayes_model,
    draws=_draws,
    tune=_tune,
    chains=_chains,
    cores=_cores,
    random_seed=42,
    return_inferencedata=True,
    target_accept=0.9,
)

# %% diagnostics

az.summary(inference_data, var_names=["beta", "delta", "gamma"]).to_csv(
    f"{_out_data}_summary.csv"
)

for var in ["beta", "delta", "gamma"]:
    az.plot_trace(inference_data, var_names=[var])
    plt.savefig(f"{_out_ql}_trace_{var}.png", bbox_inches="tight")
    plt.close("all")

for var in ["beta", "delta", "gamma"]:
    az.plot_forest(inference_data, var_names=[var], combined=True)
    plt.savefig(f"{_out_ql}_forest_{var}.png", bbox_inches="tight")
    plt.close("all")

# %% save to file 

inference_data.to_netcdf(_output)


# %%
