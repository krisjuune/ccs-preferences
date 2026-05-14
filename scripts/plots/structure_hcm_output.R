library(posterior)
library(coda)
library(tidyverse)
library(ncdf4)
library(ggplot2)
library(dplyr)

nc <- nc_open("output/inference_hybrid_choice.nc")
names(nc$var)

beta_array <- ncvar_get(nc, "posterior/beta")
dim(beta_array)
dims <- dim(beta_array)
beta_df <- expand.grid(
  level = seq_len(dims[1]),
  draw = seq_len(dims[2]),
  chain = seq_len(dims[3])
)
beta_df$beta <- as.vector(beta_array)

level_labels <- ncvar_get(nc, "constant_data/level")
beta_df$level_name <- factor(beta_df$level, labels = level_labels)

mean_beta <- beta_df |>
  group_by(level_name) |>
  summarise(mean_beta = mean(beta), .groups = "drop")

print(mean_beta)

beta_df |>
  group_by(level_name) |>
  summarise(
    mean = mean(beta),
    lower = quantile(beta, 0.11),
    upper = quantile(beta, 0.89),
    .groups = "drop"
  ) |>
  ggplot(aes(x = reorder(level_name, mean), y = mean)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  coord_flip() +
  labs(
    title = "Partworth Utilities (β)",
    x = "Attribute Level",
    y = "Utility"
  )

beta_df |>
  group_by(level_name) |>
  ggplot(aes(x = beta, fill = level_name)) +
  geom_density(alpha = 0.4) +
  geom_vline(
    xintercept = 0,
    color = "grey40",
    linetype = "dashed",
    linewidth = 0.5
  ) +
  xlim(-3.5, 3.5) +
  facet_wrap(~ level_name, scales = "free") +
  theme_classic() +
  labs(title = "Posterior Densities for β", x = "Partworth Utility", y = "Density")





delta_array <- ncvar_get(nc, "posterior/delta")
dim(delta_array)
dims <- dim(delta_array)
delta_df <- expand.grid(
  level = seq_len(dims[1]),
  draw = seq_len(dims[2]),
  chain = seq_len(dims[3])
)
delta_df$delta <- as.vector(delta_array)

level_labels <- ncvar_get(nc, "constant_data/level")
delta_df$level_name <- factor(delta_df$level, labels = level_labels)

mean_delta <- delta_df |>
  group_by(level_name) |>
  summarise(mean_delta = mean(delta), .groups = "drop")

print(mean_delta)

delta_df |>
  group_by(level_name) |>
  summarise(
    mean = mean(delta),
    lower = quantile(delta, 0.11),
    upper = quantile(delta, 0.89),
    .groups = "drop"
  ) |>
  ggplot(aes(x = reorder(level_name, mean), y = mean)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  coord_flip() +
  labs(
    title = "Framing effect",
    x = "Attribute Level",
    y = "Utility"
  )

delta_df |>
  group_by(level_name) |>
  ggplot(aes(x = delta, fill = level_name)) +
  geom_density(alpha = 0.4) +
  geom_vline(
    xintercept = 0,
    color = "grey40",
    linetype = "dashed",
    linewidth = 0.5
  ) +
  xlim(-3.5, 3.5) +
  facet_wrap(~ level_name, scales = "free") +
  theme_classic() +
  labs(title = "Posterior Densities for δ", x = "Partworth Utility", y = "Density")





gamma_array <- ncvar_get(nc, "posterior/gamma")
dim(gamma_array)
dims <- dim(gamma_array)
gamma_df <- expand.grid(
  level = seq_len(dims[1]),
  country = seq_len(dims[2]),
  draw = seq_len(dims[3]),
  chain = seq_len(dims[4])
)

gamma_df$gamma <- as.vector(gamma_array)
level_labels <- ncvar_get(nc, "constant_data/level")
gamma_df$level_name <- factor(gamma_df$level, labels = level_labels)

mean_gamma <- gamma_df |>
  group_by(level_name) |>
  summarise(mean_gamma = mean(gamma), .groups = "drop")

print(mean_gamma)




lreco_array <- ncvar_get(nc, "posterior/theta_lreco")
dim(lreco_array)
dims <- dim(lreco_array)

lreco_df <- expand.grid(
  level = seq_len(dims[1]),
  draw = seq_len(dims[2]),
  chain = seq_len(dims[3])
)

lreco_df$lreco <- as.vector(lreco_array)
level_labels <- ncvar_get(nc, "constant_data/level")
lreco_df$level_name <- factor(lreco_df$level, labels = level_labels)

mean_lreco <- lreco_df |>
  group_by(level_name) |>
  summarise(mean_lreco = mean(lreco), .groups = "drop")

print(mean_lreco)

lreco_df |>
  group_by(level_name) |>
  summarise(
    mean = mean(lreco),
    lower = quantile(lreco, 0.11),
    upper = quantile(lreco, 0.89),
    .groups = "drop"
  ) |>
  ggplot(aes(x = reorder(level_name, mean), y = mean)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  coord_flip() +
  labs(
    title = "Socio-economic (lreco) value effect",
    x = "Attribute Level",
    y = "Utility"
  )

lreco_df |>
  group_by(level_name) |>
  ggplot(aes(x = lreco, fill = level_name)) +
  geom_density(alpha = 0.4) +
  geom_vline(
    xintercept = 0,
    color = "grey40",
    linetype = "dashed",
    linewidth = 0.5
  ) +
  xlim(-3.5, 3.5) +
  facet_wrap(~ level_name, scales = "free") +
  theme_classic() +
  labs(title = "Posterior Densities for lreco", x = "Partworth Utility", y = "Density")





galtan_array <- ncvar_get(nc, "posterior/theta_galtan")
dim(galtan_array)
dims <- dim(galtan_array)

galtan_df <- expand.grid(
  level = seq_len(dims[1]),
  draw = seq_len(dims[2]),
  chain = seq_len(dims[3])
)

galtan_df$galtan <- as.vector(galtan_array)
level_labels <- ncvar_get(nc, "constant_data/level")
galtan_df$level_name <- factor(galtan_df$level, labels = level_labels)

mean_galtan <- galtan_df |>
  group_by(level_name) |>
  summarise(mean_galtan = mean(galtan), .groups = "drop")

print(mean_galtan)

galtan_df |>
  group_by(level_name) |>
  summarise(
    mean = mean(galtan),
    lower = quantile(galtan, 0.11),
    upper = quantile(galtan, 0.89),
    .groups = "drop"
  ) |>
  ggplot(aes(x = reorder(level_name, mean), y = mean)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  coord_flip() +
  labs(
    title = "Socio-economic (galtan) value effect",
    x = "Attribute Level",
    y = "Utility"
  )

galtan_df |>
  group_by(level_name) |>
  ggplot(aes(x = galtan, fill = level_name)) +
  geom_density(alpha = 0.4) +
  geom_vline(
    xintercept = 0,
    color = "grey40",
    linetype = "dashed",
    linewidth = 0.5
  ) +
  xlim(-3.5, 3.5) +
  facet_wrap(~ level_name, scales = "free") +
  theme_classic() +
  labs(title = "Posterior Densities for galtan", x = "Partworth Utility", y = "Density")
