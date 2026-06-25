# Proximity utility, shown separately for domestic and foreign CO2.
# Faceted by country; coloured with the proximity attribute colour used
# throughout the other forest plots.
library(tidyverse)
library(ggdist)
library(ggtext)
library(yaml)

config       <- read_yaml("config.yaml")
plt          <- config$plots
attr_colours <- unlist(plt$attr_colours)
base_size    <- plt$base_size
dpi_val      <- plt$dpi
slab_alpha   <- plt$slab_alpha_framing
point_size   <- plt$point_size
ci_width     <- plt$ci_width
fig_width_w  <- plt$fig_width_wide

sp_colour <- attr_colours["attr_vicinity"]  # orange matches proximity in the main plots

cond <- read_csv("output/data/posteriors_interact_conditional.csv")

country_labels <- c(
  "china"       = "China",
  "switzerland" = "Switzerland"
)

# ---- y-axis structure: domestic block, then foreign block ----

prox_order <- c(
  "attr_vicinity_abroad",
  "attr_vicinity_another region",
  "attr_vicinity_your region",
  "attr_vicinity_your municipality"
)

clean_prox_label <- function(x) {
  x |>
    str_remove("attr_vicinity_") |>
    str_to_sentence()
}

build_y_structure <- function() {
  y_levels <- c()
  y_labels <- setNames(character(0), character(0))

  for (framing in c("domestic", "foreign")) {
    header_key <- paste0("header_", framing)
    y_levels   <- c(y_levels, header_key)
    y_labels[header_key] <- paste0(
      "**Proximity utility for ", framing, " CO<sub>2</sub>**"
    )

    for (prox_l in prox_order) {
      key <- paste0(prox_l, "_", framing)
      y_levels      <- c(y_levels, key)
      y_labels[key] <- paste0("  ", clean_prox_label(prox_l))
    }
  }

  list(levels = y_levels, labels = y_labels)
}

y_struct <- build_y_structure()

# ---- prepare data ----

plot_data <- cond |>
  filter(
    interaction == "prox_source",
    country != "pooled",
    !(conditioning_level == "attr_vicinity_abroad" & framing == "domestic")
  ) |>
  mutate(
    level_key     = factor(
      paste0(conditioning_level, "_", framing),
      levels = y_struct$levels
    ),
    country_label = factor(country_labels[country], levels = unname(country_labels))
  )

# abroad/domestic is fixed at 0 by construction (reference level) — no
# posterior distribution, shown as a baseline dot instead
baseline_df <- crossing(
  tibble(level_key = factor("attr_vicinity_abroad_domestic", levels = y_struct$levels)),
  tibble(country_label = factor(unname(country_labels), levels = unname(country_labels)))
)

# ---- plot ----

ggplot(plot_data, aes(x = value, y = level_key)) +
  stat_halfeye(
    slab_alpha     = slab_alpha,
    point_size     = point_size,
    .width         = ci_width,
    point_interval = median_hdi,
    slab_colour    = NA,
    fill           = sp_colour,
    colour         = sp_colour,
    na.rm          = TRUE
  ) +
  geom_vline(xintercept = 0, linetype = "dashed",
             colour = "grey40", linewidth = 0.5) +
  geom_point(
    data  = baseline_df,
    aes(x = 0, y = level_key),
    colour = sp_colour, size = point_size, shape = 19, na.rm = TRUE,
    inherit.aes = FALSE
  ) +
  scale_y_discrete(
    labels = y_struct$labels,
    limits = rev(y_struct$levels),
    drop   = FALSE
  ) +
  facet_wrap(~ country_label, ncol = 2) +
  labs(x = "Proximity utility (β + γ)", y = NULL) +
  theme_classic(base_size = base_size) +
  theme(
    axis.text.y        = element_markdown(lineheight = 1.2),
    panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.4),
    strip.text         = element_text(face = "bold", size = base_size),
    strip.background   = element_blank(),
    plot.margin        = margin(10, 20, 10, 10)
  )

ggsave("output/plots/interact_proximity.png",
       width = fig_width_w, height = 6, dpi = dpi_val, bg = "white")
