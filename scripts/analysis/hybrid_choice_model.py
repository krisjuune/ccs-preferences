import pymc as pm
import pandas as pd
import arviz as az
import numpy as np

# %% pymc bug workaround

import pytensor

pytensor.config.cxx = "/usr/bin/clang++"

# %%

df = (
    pd.read_csv("~/Downloads/data_translated_ch.csv")
    .assign(country="CH")
    .head(12 * 100)
)

# %% define attributes and baselines

attributes = [
    # "attr_engagement",
    "attr_vicinity",
    "attr_industry",
    "attr_costs",
    # "attr_reason",
    "attr_source_purpose",
]

baseline_dict = {
    # "attr_engagement": "inform",
    "attr_vicinity": "abroad",
    "attr_industry": "waste incineration",
    "attr_costs": "taxpayer",
    # "attr_reason": "sparsely-populated",
    "attr_source_purpose": "domestic",
}

# %% define dummies

# reorder each attribute column by making it categorical with the baseline first
for attr in attributes:
    baseline = baseline_dict[attr]
    df[attr] = pd.Categorical(
        df[attr],
        categories=[baseline]
        + [level for level in df[attr].unique() if level != baseline],
        ordered=True,
    )

# generate dummies with columns in the correct order
# the next line is the important change
dummies = pd.get_dummies(df[attributes], drop_first=True)

# reorder columns to place baseline first for each attribute
ordered_columns = []
for attr in attributes:
    # Collect the columns related to the attribute and put baseline first
    attr_columns = [col for col in dummies.columns if col.startswith(attr)]
    baseline_column = f"{attr}_{baseline_dict[attr]}"
    # ordered_columns.append(baseline_column)
    ordered_columns.extend([col for col in attr_columns if col != baseline_column])

# reorder dummies according to ordered columns list
dummies = dummies[ordered_columns]
dummies = dummies.loc[:, ~dummies.columns.duplicated()]

df["framing"] = df["framing"].astype("category")
df["country"] = df["country"].astype("category")

# add dimensions
unique_individuals = df["id"].unique()
coords = {
    "level": dummies.columns.tolist(),
    "task": np.arange(len(df) // 2),
    "framing": df["framing"].cat.categories,
    "country": df["country"].cat.categories,
    "individual": unique_individuals,
}

id_to_index = {id_: i for i, id_ in enumerate(unique_individuals)}
df["individual_idx"] = df["id"].map(id_to_index)
individual_idx = df.loc[df.package == 1, "individual_idx"].values

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
    # lreco_latent = pm.Normal("lreco_latent", mu=0, sigma=1, dims="individual")
    # galtan_latent = pm.Normal("galtan_latent", mu=0, sigma=1, dims="individual")
    # socio_ecol_latent = pm.Normal("socio_ecol_latent", mu=0, sigma=1, dims="individual")

    # noise in observed scores
    # likert_sigma = 0.1

    # Observed variables as continuous (normalized to 0–1)
    # pm.Normal("lreco_1", mu=lreco_latent[individual_idx], sigma=likert_sigma,
    #           observed=df.loc[df.package == 1, "lreco_1"].values)

    # pm.Normal("lreco_2", mu=lreco_latent[individual_idx], sigma=likert_sigma,
    #           observed=df.loc[df.package == 1, "lreco_2"].values)

    # pm.Normal("lreco_3", mu=lreco_latent[individual_idx], sigma=likert_sigma,
    #           observed=df.loc[df.package == 1, "lreco_3"].values)

    # pm.Normal("galtan_1", mu=galtan_latent[individual_idx], sigma=likert_sigma,
    #           observed=df.loc[df.package == 1, "galtan_1"].values)

    # pm.Normal("galtan_2", mu=galtan_latent[individual_idx], sigma=likert_sigma,
    #           observed=df.loc[df.package == 1, "galtan_2"].values)

    # pm.Normal("galtan_3", mu=galtan_latent[individual_idx], sigma=likert_sigma,
    #           observed=df.loc[df.package == 1, "galtan_3"].values)

    # pm.Normal("socio_ecological_1", mu=socio_ecol_latent[individual_idx], sigma=likert_sigma,
    #           observed=df.loc[df.package == 1, "socio_ecological_1"].values)

    # pm.Normal("socio_ecological_2", mu=socio_ecol_latent[individual_idx], sigma=likert_sigma,
    #           observed=df.loc[df.package == 1, "socio_ecological_2"].values)

    # pm.Normal("socio_ecological_3", mu=socio_ecol_latent[individual_idx], sigma=likert_sigma,
    #           observed=df.loc[df.package == 1, "socio_ecological_3"].values)

    # get framing and country codes
    f = pm.Data("f", df.loc[df.package == 1, "framing"].cat.codes.values, dims="task")
    f = f.astype("float32")
    country_idx = df.loc[df.package == 1, "country"].cat.codes.values
    c = pm.Data("c", country_idx, dims="task")

    # observed choices
    observed_choice_left = pm.Data(
        "observed_choice_left", df.loc[df.package == 1, "chosen"].values, dims="task"
    )

    # attribute dummies
    attribute_levels_left = pm.Data(
        "attribute_levels_left",
        dummies[df.package == 1].values * 1,
        dims=["task", "level"],
    )

    attribute_levels_right = pm.Data(
        "attribute_levels_right",
        dummies[df.package == 2].values * 1,
        dims=["task", "level"],
    )

    # gamma coefficients: how much each latent trait moderates framing effects
    theta_lreco = pm.Normal("theta_lreco", mu=0, sigma=1, dims="level")
    theta_galtan = pm.Normal("theta_galtan", mu=0, sigma=1, dims="level")
    theta_ecol = pm.Normal("theta_ecol", mu=0, sigma=1, dims="level")

    # choice model: main effects
    beta = pm.Normal("beta", mu=0, sigma=2, dims="level")

    # framing-specific shift
    delta = pm.Normal("delta", mu=0, sigma=1, dims="level")

    # country effect
    gamma = pm.Normal("gamma", mu=0, sigma=1, dims=["country", "level"])

    beta_modulated = (
        beta + delta * f[:, None] + gamma[c, :]
        # + theta_lreco * lreco_latent[individual_idx][:, None]
        # + theta_galtan * galtan_latent[individual_idx][:, None]
        # + theta_ecol * socio_ecol_latent[individual_idx][:, None]
    )

    # get utilities
    utility_left = pm.Deterministic(
        "utility_left",
        pm.math.sum(attribute_levels_left * beta_modulated, axis=1),
        dims="task",
    )

    utility_right = pm.Deterministic(
        "utility_right",
        pm.math.sum(attribute_levels_right * beta_modulated, axis=1),
        dims="task",
    )

    # logit probability
    prob_choice_left = pm.Deterministic(
        "probability_choice_left",
        pm.math.exp(utility_left)
        / (pm.math.exp(utility_left) + pm.math.exp(utility_right)),
        dims="task",
    )

    # likelihood
    pm.Bernoulli(
        "choice_distribution", p=prob_choice_left, observed=observed_choice_left
    )

# %% get priors

priors = pm.sample_prior_predictive(
    samples=1000,
    model=hcm_model,
    random_seed=42,
)

# %% check priors

priors.prior
priors.prior.keys()
az.summary(
    priors,
    var_names=[
        "beta",
        "gamma",
        "delta",
        "theta_lreco",
    ],
)

# %% test model

# with hcm_model:
#    approx = pm.fit(n=10000, method="advi")
#    trace = approx.sample(1000)

# %% run model (3 to 5 hours)

# run model with MCMC with 1000 draws, 500 tune samples, and 4 chains on 6 cores
inference_data = pm.sample(
    model=hcm_model,
    draws=1000,
    tune=500,
    chains=4,
    cores=4,  # more cores than chains has no effect
    random_seed=42,
    return_inferencedata=True,
    target_accept=0.9,
)

# %% diagnostics

az.summary(
    inference_data,
    var_names=["beta", "delta", "gamma", "theta_lreco", "theta_galtan", "theta_ecol"],
)

az.plot_trace(
    inference_data,
    var_names=["beta", "delta", "gamma", "theta_lreco", "theta_galtan", "theta_ecol"],
)

# az.plot_dist(inference_data, var_names=[
#     "beta",
#     "delta",
#     "theta_lreco",
#     "theta_galtan",
#     "theta_ecol"
#     ])

az.plot_forest(inference_data, var_names=["beta"], combined=True)
az.plot_forest(inference_data, var_names=["delta"], combined=True)
az.plot_forest(inference_data, var_names=["gamma"], combined=True)
az.plot_forest(inference_data, var_names=["theta_lreco"], combined=True)
az.plot_forest(inference_data, var_names=["theta_galtan"], combined=True)
az.plot_forest(inference_data, var_names=["theta_ecol"], combined=True)

# %% save to netcdf

inference_data.to_netcdf("output/inference_hybrid_choice.nc")

# %%
