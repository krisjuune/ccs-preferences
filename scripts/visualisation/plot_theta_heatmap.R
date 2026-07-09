library(tidyverse)
library(ggtext)
library(yaml)

# ---- coding setting ----

config    <- read_yaml("config.yaml")
coding    <- config$coding
plt       <- config$plots
ci_width  <- plt$ci_width
base_size <- plt$base_size
fig_width <- plt$fig_width
dpi_val   <- plt$dpi
div_low   <- plt$diverging_colours$low
div_mid   <- plt$diverging_colours$mid
div_high  <- plt$diverging_colours$high

baseline_level_names <- c(
  "attr_engagement_inform",
  "attr_vicinity_abroad",
  "attr_industry_waste incineration",
  "attr_costs_taxpayer",
  "attr_reason_sparsely-populated",
  "attr_source_purpose_domestic"
)

# ---- load data ----

theta <- read_csv("output/data/posteriors_theta.csv")

# ---- attribute setup (shared with other plot scripts) ----

attr_order <- c(
  "attr_vicinity",
  "attr_source_purpose",
  "attr_industry",
  "attr_costs",
  "attr_reason",
  "attr_engagement"
)

attr_display <- c(
  "attr_engagement"     = "Engagement",
  "attr_vicinity"       = "Proximity",
  "attr_industry"       = "Industry",
  "attr_costs"          = "Cost responsibility",
  "attr_reason"         = "Location reason",
  "attr_source_purpose" = "Source / Purpose"
)

level_display <- c(
  "attr_engagement_inform"                     = "Inform",
  "attr_engagement_consult"                    = "Consult",
  "attr_engagement_vote"                       = "Vote",
  "attr_vicinity_abroad"                       = "Abroad",
  "attr_vicinity_another region"               = "Another region",
  "attr_vicinity_your region"                  = "Your region",
  "attr_vicinity_your municipality"            = "Your municipality",
  "attr_industry_waste incineration"           = "Waste incineration",
  "attr_industry_metal and cement production"  = "Metal & cement",
  "attr_industry_gas with CCS"                 = "Gas with CCS",
  "attr_costs_taxpayer"                        = "Taxpayer",
  "attr_costs_polluting industry"              = "Polluting industry",
  "attr_reason_sparsely-populated"             = "Sparsely populated",
  "attr_reason_close to source"                = "Close to source",
  "attr_reason_cost-efficient"                 = "Cost-efficient",
  "attr_source_purpose_domestic"               = "Domestic",
  "attr_source_purpose_foreign"                = "Foreign"
)

get_attribute <- function(level) {
  for (attr in attr_order) {
    if (startsWith(level, attr)) return(attr)
  }
  NA_character_
}

build_y_structure <- function() {
  y_levels <- c()
  y_labels <- setNames(character(0), character(0))
  for (attr in attr_order) {
    header_key <- paste0("header_", attr)
    y_levels   <- c(y_levels, header_key)
    y_labels[header_key] <- paste0("**", attr_display[[attr]], "**")
    attr_levels <- names(level_display)[startsWith(names(level_display), attr)]
    for (lvl in attr_levels) {
      y_levels      <- c(y_levels, lvl)
      y_labels[lvl] <- paste0("  ", level_display[[lvl]])
    }
  }
  list(levels = y_levels, labels = y_labels)
}

y_struct <- build_y_structure()

# ---- value dimension labels and ordering ----

dim_order  <- c("lreco", "galtan", "ecol")
dim_labels <- c(
  "lreco"  = "Socio-\neconomic",
  "galtan" = "Socio-\ncultural",
  "ecol"   = "Socio-\necological"
)

# ---- summarise posteriors ----

theta_summary <- theta |>
  filter(model == "hybrid") |>
  filter(level %in% names(level_display)) |>
  group_by(dim, level) |>
  summarise(
    mean  = mean(value),
    lower = quantile(value, (1 - ci_width) / 2),
    upper = quantile(value, 1 - (1 - ci_width) / 2),
    .groups = "drop"
  ) |>
  mutate(
    credible   = lower > 0 | upper < 0,
    tile_alpha = if_else(credible, 1.0, 0.3),
    dim_label  = factor(dim_labels[dim], levels = unname(dim_labels)),
    level_base = factor(level, levels = y_struct$levels)
  )

# add baseline rows at 0 when using reference_level coding
if (coding == "reference_level") {
  baseline_rows <- crossing(
    tibble(level = baseline_level_names[baseline_level_names %in% names(level_display)]),
    tibble(dim = dim_order)
  ) |>
    mutate(
      mean       = 0, lower = 0, upper = 0,
      credible   = FALSE, tile_alpha = 0.3,
      dim_label  = factor(dim_labels[dim], levels = unname(dim_labels)),
      level_base = factor(level, levels = y_struct$levels)
    )
  theta_summary <- bind_rows(theta_summary, baseline_rows)
}

# ---- plot ----
# header rows have no tile data — scale_y_discrete(drop=FALSE) keeps them
# as empty rows, providing visual grouping without needing geom_tile entries

ggplot(theta_summary, aes(x = dim_label, y = level_base)) +
  geom_tile(
    aes(fill = mean, alpha = tile_alpha),
    colour    = "white",
    linewidth = 0.8
  ) +
  # label credible tiles with the rounded posterior mean
  geom_text(
    data = filter(theta_summary, credible),
    aes(label = sprintf("%.2f", mean)),
    size   = 3.5,
    colour = "white",
    fontface = "bold"
  ) +
  scale_fill_gradient2(
    low      = div_low,
    mid      = div_mid,
    high     = div_high,
    midpoint = 0,
    name     = "θ (posterior mean)"
  ) +
  scale_alpha_identity() +
  scale_y_discrete(
    labels = y_struct$labels,
    limits = rev(y_struct$levels),
    drop   = FALSE
  ) +
  labs(x = NULL, y = NULL) +
  theme_classic(base_size = base_size) +
  theme(
    axis.text.y     = element_markdown(lineheight = 1.2),
    axis.text.x     = element_text(face = "bold", size = base_size - 1),
    axis.ticks.x    = element_blank(),
    legend.position = "right",
    plot.margin     = margin(10, 20, 10, 10)
  )

ggsave(
  "output/plots/supp_figs/theta_heatmap.png",
  width = fig_width, height = 10, dpi = dpi_val, bg = "white"
)
