library(tidyverse)
library(ggdist)
library(yaml)

config        <- read_yaml("config.yaml")
plt           <- config$plots
value_colours <- unlist(plt$value_colours)
div_colours   <- unlist(plt$diverging_colours)
slab_alpha   <- plt$slab_alpha
slab_alpha_f <- plt$slab_alpha_framing
point_size   <- plt$point_size
point_alpha  <- plt$point_alpha
ci_width     <- plt$ci_width
base_size    <- plt$base_size
fig_width    <- plt$fig_width
fig_width_w  <- plt$fig_width_wide
dpi_val      <- plt$dpi

# ---- load data ----

pred_data <- read_csv(
  "output/predictions/prediction_scenarios.csv",
  show_col_types = FALSE
) |>
  mutate(
    country_label = factor(
      recode(country, china = "China", switzerland = "Switzerland"),
      levels = c("China", "Switzerland")
    )
  )

country_order <- rev(levels(pred_data$country_label))

dim_display <- c(
  "lreco"  = "Left-right economic",
  "galtan" = "GAL-TAN (socio-cultural)",
  "ecol"   = "Socio-ecological"
)

# One base hue per value dimension (consistent with plot_value_distributions.R):
# lreco = blue, galtan = orange, ecol = green. Higher z is always plotted
# opaque (full slab_alpha) and lower z faded (slab_alpha_framing) — this
# already lines up with "more left" (lreco), "more GAL/alternative"
# (galtan), and "more ecological" (ecol) for this model's latent-variable
# sign convention, so no per-dimension flipping is needed. Points/intervals
# use the same full-saturation colour and point_alpha at both z-levels —
# only the slab (density) fades, so the summary stats stay legible.
dim_base_colour <- c(
  "lreco"  = value_colours[["lreco"]],
  "galtan" = value_colours[["galtan"]],
  "ecol"   = value_colours[["socio_ecol"]]
)

comparison_labels <- c(
  "vs_abroad" = "Local siting vs siting abroad",
  "vs_region" = "Local siting vs siting in another region"
)
comparison_order <- unname(comparison_labels)

comparison_label_from_scenario <- function(scenario) {
  factor(
    if_else(str_detect(scenario, "_vs_abroad$"),
            comparison_labels[["vs_abroad"]], comparison_labels[["vs_region"]]),
    levels = comparison_order
  )
}

# ---- shared halfeye-by-z-level plot (used by H2 and H3 figures) ----
# Two overlapping distributions per row: low z (faded slab) vs high z
# (opaque slab), same full-saturation colour and point_alpha for both so
# the point/interval summary is never faded. `data` must have a `dim`
# column (one of "lreco", "galtan", "ecol") so each row's hue is looked up
# from dim_base_colour — this is what lets H3's multi-dimension figures
# show a different hue per facet.
#
# The real fill/colour is rendered via scale_*_identity() (no legend,
# since a single legend can't represent multiple facet hues at once — the
# facet strip / caption explain hue). A separate dummy layer with a fixed
# neutral grey provides the Low/High legend, which is about opacity only.

plot_by_z <- function(data, x_lab, facet_var = NULL, facet_ncol = 1, legend_title = NULL,
                       facet_rows = NULL, facet_cols = NULL) {
  data <- data |>
    mutate(fill_hex = dim_base_colour[dim])

  legend_data <- tibble(
    z_label = factor(c("Low (z = -1.5)", "High (z = +1.5)"),
                      levels = c("Low (z = -1.5)", "High (z = +1.5)")),
    x = -1,
    y = country_order[1]
  )

  p <- ggplot(
    data,
    aes(x = prob, y = country_label, group = interaction(country_label, z_level))
  ) +
    stat_halfeye(
      data           = \(d) filter(d, z_level == "low"),
      mapping        = aes(fill = fill_hex, colour = fill_hex),
      slab_alpha     = slab_alpha_f,
      scale          = 0.45,
      point_alpha    = point_alpha,
      interval_alpha = point_alpha,
      point_size     = point_size,
      .width         = ci_width,
      point_interval = median_hdi,
      slab_colour    = NA
    ) +
    stat_halfeye(
      data           = \(d) filter(d, z_level == "high"),
      mapping        = aes(fill = fill_hex, colour = fill_hex),
      slab_alpha     = slab_alpha,
      scale          = 0.45,
      point_alpha    = point_alpha,
      interval_alpha = point_alpha,
      point_size     = point_size,
      .width         = ci_width,
      point_interval = median_hdi,
      slab_colour    = NA
    ) +
    geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
    scale_fill_identity(guide = "none") +
    scale_colour_identity(guide = "none") +
    geom_point(
      data        = legend_data,
      mapping     = aes(x = x, y = y, alpha = z_label),
      inherit.aes = FALSE,
      colour      = "grey30",
      size        = point_size,
      na.rm       = TRUE
    ) +
    scale_alpha_manual(
      name   = legend_title,
      values = c("Low (z = -1.5)" = slab_alpha_f, "High (z = +1.5)" = 1),
      guide  = guide_legend(override.aes = list(size = 5))
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      labels = scales::percent_format(accuracy = 1),
      expand = expansion(mult = c(0.01, 0.05))
    ) +
    scale_y_discrete(limits = country_order) +
    labs(x = x_lab, y = NULL) +
    theme_classic(base_size = base_size) +
    theme(
      panel.grid.major.x   = element_line(colour = "grey92", linewidth = 0.4),
      strip.text           = element_text(face = "bold", size = base_size),
      strip.background     = element_blank(),
      legend.position      = "bottom",
      legend.justification = "center",
      plot.margin          = margin(10, 20, 10, 10)
    )

  if (!is.null(facet_rows) || !is.null(facet_cols)) {
    p <- p + facet_grid(
      rows = if (!is.null(facet_rows)) vars(.data[[facet_rows]]) else NULL,
      cols = if (!is.null(facet_cols)) vars(.data[[facet_cols]]) else NULL
    )
  } else if (!is.null(facet_var)) {
    p <- p + facet_wrap(vars(.data[[facet_var]]), ncol = facet_ncol)
  }
  p
}

# ==== H2: proximity x ecological orientation ====
# Pooled "locally sited" quantity (your municipality or your region, equal
# weight), same as H1. Two panels: vs. abroad and vs. another region.

h2_data <- pred_data |>
  filter(scenario %in% c("proximity_ecol_vs_abroad", "proximity_ecol_vs_region")) |>
  mutate(
    dim               = "ecol",
    comparison_label  = comparison_label_from_scenario(scenario)
  )

plot_by_z(h2_data, "Predicted probability of local siting",
          facet_var = "comparison_label", facet_ncol = 2,
          legend_title = "Ecological orientation")

ggsave(
  "output/plots/predictions_proximity.png",
  width = fig_width_w, height = 4, dpi = dpi_val, bg = "white"
)

# ==== H1: inclusive design (inclusive / neutral / extractive bundle) ====
# No z split — single pooled P per country per bundle, at the population
# mean. "Locally sited" pools your municipality / your region (see
# predict_scenarios.py's local_components()). Two panels: vs. abroad
# (left) and vs. another region (right). Bundle attribute lists are in the
# figure caption, not the legend.

bundle_colours <- c(
  "Extractive" = div_colours[["low"]],
  "Neutral"    = div_colours[["neutral"]],
  "Inclusive"  = div_colours[["high"]]
)
bundle_order <- names(bundle_colours)

h1_data <- pred_data |>
  filter(hypothesis == "H1", z_level == "fixed") |>
  mutate(
    bundle = factor(
      case_when(
        str_detect(scenario, "^design_inclusive")  ~ "Inclusive",
        str_detect(scenario, "^design_neutral")    ~ "Neutral",
        str_detect(scenario, "^design_extractive") ~ "Extractive"
      ),
      levels = bundle_order
    ),
    comparison_label = comparison_label_from_scenario(scenario)
  )

ggplot(
  h1_data,
  aes(x = prob, y = country_label, fill = bundle, colour = bundle)
) +
  stat_halfeye(
    scale          = 0.55,
    slab_alpha     = slab_alpha,
    point_alpha    = point_alpha,
    interval_alpha = point_alpha,
    point_size     = point_size,
    .width         = ci_width,
    point_interval = median_hdi,
    slab_colour    = NA,
    position       = position_dodgejust(width = 0.9)
  ) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
  scale_fill_manual(name = NULL, values = bundle_colours, breaks = bundle_order) +
  scale_colour_manual(name = NULL, values = bundle_colours, breaks = bundle_order) +
  scale_x_continuous(
    limits = c(0, 1),
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0.01, 0.05))
  ) +
  scale_y_discrete(limits = country_order) +
  facet_wrap(vars(comparison_label), ncol = 2) +
  labs(x = "Predicted probability of local siting", y = NULL) +
  theme_classic(base_size = base_size) +
  theme(
    panel.grid.major.x   = element_line(colour = "grey92", linewidth = 0.4),
    strip.text           = element_text(face = "bold", size = base_size),
    strip.background     = element_blank(),
    legend.position      = "bottom",
    legend.justification = "center",
    plot.margin          = margin(10, 20, 10, 10)
  )

ggsave(
  "output/plots/predictions_design.png",
  width = fig_width_w, height = 5, dpi = dpi_val, bg = "white"
)

# ==== H3: domain-matching — cost responsibility & CO2 origin x all three ====
# value dimensions, combined into one grid (rows = value dimension, columns
# = attribute). Cost responsibility now has all three dimensions tested
# (including costs_ecol, a specificity check: H3 predicts ecol should NOT
# move cost responsibility), so it lines up with source/purpose's three
# dimensions for a direct side-by-side comparison — the domain-matching
# pattern (costs moves with lreco/galtan but not ecol; source moves with
# all three) is visible at a glance across the grid.

h3_data <- bind_rows(
  pred_data |>
    filter(scenario %in% c("costs_lreco", "costs_galtan", "costs_ecol")) |>
    mutate(dim = str_remove(scenario, "^costs_"), attribute = "Cost responsibility"),
  pred_data |>
    filter(scenario %in% c("source_lreco", "source_galtan", "source_ecol")) |>
    mutate(dim = str_remove(scenario, "^source_"), attribute = "CO2 origin")
) |>
  mutate(
    dim_label = factor(dim_display[dim], levels = unname(dim_display[c("lreco", "galtan", "ecol")])),
    attribute = factor(attribute, levels = c("Cost responsibility", "CO2 origin"))
  )

plot_by_z(h3_data, "Predicted probability of choosing the value-aligned level",
          facet_rows = "dim_label", facet_cols = "attribute", legend_title = "Value level") +
  theme(plot.margin = margin(10, 30, 10, 10))

ggsave(
  "output/plots/predictions_values.png",
  width = fig_width_w, height = 10, dpi = dpi_val, bg = "white"
)
