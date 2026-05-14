import pymc as pm 
import pandas as pd
import arviz as az
import numpy as np
import xarray as xr

# %% pymc bug workaround

import pytensor
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

# generate dummies with columns in the correct order
dummies = pd.get_dummies(df[attributes], drop_first=False)

# reorder columns to place baseline first for each attribute
ordered_columns = []
for attr in attributes:
    # Collect the columns related to the attribute and put baseline first
    attr_columns = [col for col in dummies.columns if col.startswith(attr)]
    baseline_column = f"{attr}_{baseline_dict[attr]}"
    ordered_columns.append(baseline_column)
    ordered_columns.extend([col for col in attr_columns if col != baseline_column])

# reorder dummies according to ordered columns list
dummies = dummies[ordered_columns]
dummies = dummies.loc[:, ~dummies.columns.duplicated()]

coords = {
    "level": dummies.columns.values,
    "task": np.arange(len(df) // 2),
}


# %% define model

with pm.Model(coords=coords) as simple_model:
    # main effect of attribute levels (one beta per dummy level)
    beta = pm.Normal("beta", mu=0, sigma=2, dims="level")  # shape: (n_levels,)

    # observed choices (left chosen = 1, right chosen = 0) for tasks
    observed_choice_left = pm.Data(
        "observed_choice_left",
        df.loc[df.package == 1, "chosen"].values,
        dims="task"
    )

    # attribute dummies for left and right alternatives (shape: n_tasks x n_levels)
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

    # compute utilities (dot product of dummies and beta)
    utility_left = pm.Deterministic(
        "utility_left",
        pm.math.sum(attribute_levels_left * beta, axis=1),
        dims="task"
    )

    utility_right = pm.Deterministic(
        "utility_right",
        pm.math.sum(attribute_levels_right * beta, axis=1),
        dims="task"
    )

    # probability left chosen: sigmoid(utility_left - utility_right) for stability
    probability_choice_left = pm.Deterministic(
        "probability_choice_left",
        pm.math.sigmoid(utility_left - utility_right),
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
    samples = 100, 
    model = simple_model, 
    random_seed = 42, 
)

# %% check priors

priors.prior
az.summary(priors, var_names = ["beta"])
# az.plot_forest(priors, var_names=["delta"], combined=True)
# delta_draws = priors.prior["delta"]


# %%

# run model with MCMC with 1000 draws, 500 tune samples, and 4 chains on 6 cores
inference_data = pm.sample(
    model = simple_model, 
    draws = 250, 
    tune = 125, 
    chains = 4,
    cores = 6, 
    random_seed = 42, 
    return_inferencedata = True, 
    target_accept = 0.9
)

# %% diagnostics

az.summary(inference_data, var_names=["beta"])
az.plot_trace(inference_data, var_names=["beta"])
az.plot_forest(inference_data, var_names=["beta"], combined=True)

# %% save to file 

inference_data.to_netcdf("output/inference_choice.nc")


# %%
