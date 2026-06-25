library(tidyverse)
library(yaml)

# ---- settings ----

config      <- read_yaml("config.yaml")
plt         <- config$plots
base_size   <- plt$base_size
fig_width_w <- plt$fig_width_wide
dpi_val     <- plt$dpi
div_low     <- plt$diverging_colours$low
div_mid     <- plt$diverging_colours$mid
div_high    <- plt$diverging_colours$high

var_order <- c(
  "Left-right economic", "GAL-TAN", "Socio-ecological",
  "Age", "Education", "Income", "Gender (female = 1)"
)

# ---- load and reshape ----

read_corr <- function(path, country_label) {
  read_csv(path, show_col_types = FALSE) |>
    rename(var1 = 1) |>
    pivot_longer(-var1, names_to = "var2", values_to = "value") |>
    mutate(
      country = country_label,
      var1 = factor(var1, levels = var_order),
      var2 = factor(var2, levels = var_order)
    )
}

corr_data <- bind_rows(
  read_corr("output/tables/value_correlations_ch.csv", "Switzerland"),
  read_corr("output/tables/value_correlations_cn.csv", "China")
) |>
  mutate(country = factor(country, levels = c("China", "Switzerland")))

# ---- plot ----

ggplot(corr_data, aes(x = var2, y = var1, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.2f", value)), size = 3.2, colour = "black") +
  scale_fill_gradient2(
    low = div_low, mid = div_mid, high = div_high,
    midpoint = 0, limits = c(-1, 1), name = "Pearson r"
  ) +
  scale_y_discrete(limits = rev(var_order)) +
  facet_wrap(~ country, ncol = 2) +
  labs(x = NULL, y = NULL) +
  theme_classic(base_size = base_size) +
  theme(
    axis.text.x       = element_text(angle = 45, hjust = 1),
    strip.text        = element_text(face = "bold", size = base_size),
    strip.background  = element_blank(),
    panel.grid        = element_blank(),
    legend.position   = "right",
    plot.margin       = margin(10, 20, 10, 10)
  )

ggsave(
  "output/plots/supp_figs/value_correlations.png",
  width = fig_width_w, height = 6, dpi = dpi_val, bg = "white"
)
