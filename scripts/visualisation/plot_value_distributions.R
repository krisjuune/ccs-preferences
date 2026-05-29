library(tidyverse)
library(yaml)

# ---- settings ----

config    <- read_yaml("config.yaml")
plt       <- config$plots
base_size <- plt$base_size
dpi_val   <- plt$dpi

# ---- load data ----

values <- read_csv("data/data_values_ch_cn.csv") |>
  select(id, country, lreco, galtan, socio_ecol) |>
  distinct() |>
  drop_na()

# ---- reshape ----

dim_labels <- c(
  "lreco"      = "Left-right economic",
  "galtan"     = "GAL-TAN",
  "socio_ecol" = "Socio-ecological"
)

country_labels <- c(
  "china"       = "China",
  "switzerland" = "Switzerland"
)

dim_colours <- c(
  "Left-right economic" = "#4e79a7",
  "GAL-TAN"             = "#f28e2b",
  "Socio-ecological"    = "#59a14f"
)

values_long <- values |>
  pivot_longer(
    cols      = c(lreco, galtan, socio_ecol),
    names_to  = "dimension",
    values_to = "score"
  ) |>
  mutate(
    country_label = factor(country_labels[country], levels = unname(country_labels)),
    dim_label     = factor(dim_labels[dimension],   levels = unname(dim_labels))
  )

# ---- plot ----

ggplot(values_long, aes(x = score, fill = dim_label, colour = dim_label)) +
  geom_density(alpha = 0.35, linewidth = 0.8) +
  geom_vline(xintercept = 1.5, linetype = "dashed", colour = "grey40", linewidth = 0.4) +
  facet_wrap(~ country_label, ncol = 2) +
  scale_fill_manual(values = dim_colours, name = NULL) +
  scale_colour_manual(values = dim_colours, name = NULL) +
  labs(x = "Value index (sum of normalised items, range 0–3)", y = "Density") +
  theme_classic(base_size = base_size) +
  theme(
    strip.text         = element_text(face = "bold", size = base_size),
    strip.background   = element_blank(),
    legend.position    = "bottom",
    legend.justification = "center"
  )

ggsave(
  "output/plots/value_distributions.png",
  width = 10, height = 5, dpi = dpi_val, bg = "white"
)
