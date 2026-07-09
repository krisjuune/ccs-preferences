library(tidyverse)
library(ggdist)
library(ggtext)
library(yaml)

# ---- coding setting ----

config       <- read_yaml("config.yaml")
coding       <- config$coding
plt          <- config$plots
attr_colours <- unlist(plt$attr_colours)
slab_alpha   <- plt$slab_alpha
slab_alpha_f <- plt$slab_alpha_framing
point_size   <- plt$point_size
point_alpha  <- plt$point_alpha
ci_width     <- plt$ci_width
base_size    <- plt$base_size
fig_width_w  <- plt$fig_width_wide
dpi_val      <- plt$dpi
sp_light     <- plt$sp_source_light

baseline_level_names <- c(
  "attr_engagement_inform",
  "attr_vicinity_abroad",
  "attr_industry_waste incineration",
  "attr_costs_taxpayer",
  "attr_reason_sparsely-populated",
  "attr_source_purpose_domestic"
)

# ---- load data ----

country_data <- read_csv("output/data/posteriors_country_total.csv")

# ---- attribute ordering and display ----

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

# ---- level labels (keyed on base level name, no framing suffix) ----

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

# ---- y-axis structure ----

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

# ---- country labels for facets ----

country_labels <- c(
  "china"       = "China",
  "switzerland" = "Switzerland"
)

# ---- prepare data ----

plot_data <- country_data |>
  filter(model == "hybrid") |>
  filter(
    (framing == "average" & !grepl("source_purpose", level)) |
    (framing %in% c("source", "purpose") & grepl("source_purpose", level))
  ) |>
  mutate(
    level_base    = gsub("_(source|purpose)$", "", level),
    attribute     = sapply(level_base, get_attribute),
    country_label = country_labels[country]
  ) |>
  filter(level_base %in% names(level_display)) |>
  mutate(level_base = factor(level_base, levels = y_struct$levels))

header_rows <- crossing(
  tibble(
    level_base = factor(
      grep("^header_", y_struct$levels, value = TRUE),
      levels = y_struct$levels
    ),
    value     = NA_real_,
    model     = "hybrid",
    framing   = "average",
    attribute = NA_character_,
    chain     = 1L,
    draw      = 1L
  ),
  country       = names(country_labels),
  country_label = unname(country_labels)
)

plot_data <- bind_rows(plot_data, header_rows) |>
  mutate(
    level_base = factor(level_base, levels = y_struct$levels),
    fill_group = case_when(
      is.na(attribute)  ~ NA_character_,
      framing == "purpose" & grepl("source_purpose", as.character(level_base)) ~ "sp_purpose",
      grepl("source_purpose", as.character(level_base)) ~ "sp_source",
      TRUE ~ attribute
    )
  )

# legend setup — identical to plot_partworths.R
row1_breaks <- c("attr_vicinity", "sp_source", "sp_purpose")
row1_values <- c(
  "attr_vicinity" = attr_colours[["attr_vicinity"]],
  "sp_source"     = attr_colours[["attr_source_purpose"]],
  "sp_purpose"    = "#b07aa1"   # hardcoded: must match sp_source exactly for legend override.aes to render correctly
)
row1_labels <- c(
  "attr_vicinity" = "Proximity",
  "sp_source"     = "Source / Purpose (purpose framing)",
  "sp_purpose"    = "Source / Purpose (source framing)"
)
row2_breaks <- c("attr_industry", "attr_costs", "attr_reason", "attr_engagement")
row2_values <- attr_colours[row2_breaks]
row2_labels <- c(
  "attr_industry"   = "Industry",
  "attr_costs"      = "Cost responsibility",
  "attr_reason"     = "Location reason",
  "attr_engagement" = "Engagement"
)

# ---- plot ----

# baseline rows (reference_level only): shown as a dot at x=0 per country
baseline_df <- if (coding == "reference_level") {
  crossing(
    tibble(
      level_base = factor(baseline_level_names, levels = y_struct$levels),
      fill_group = if_else(
        grepl("source_purpose", baseline_level_names),
        "sp_source",
        sapply(baseline_level_names, get_attribute)
      )
    ),
    tibble(
      country       = names(country_labels),
      country_label = unname(country_labels)
    )
  )
} else NULL

ggplot(
  plot_data,
  aes(
    x      = value,
    y      = level_base,
    group  = interaction(level_base, framing),
    fill   = fill_group,
    colour = fill_group
  )
) +
  stat_halfeye(
    data           = \(d) filter(d, framing != "purpose"),
    slab_alpha     = slab_alpha_f,
    point_alpha    = point_alpha,
    interval_alpha = point_alpha,
    point_size     = point_size,
    .width         = ci_width,
    point_interval = median_hdi,
    slab_colour    = NA,
    na.rm          = TRUE
  ) +
  stat_halfeye(
    data           = \(d) filter(d, framing == "purpose"),
    slab_alpha     = slab_alpha,
    point_alpha    = point_alpha,
    interval_alpha = point_alpha,
    point_size     = point_size,
    .width         = ci_width,
    point_interval = median_hdi,
    slab_colour    = NA,
    na.rm          = TRUE
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.5) +
  {if (!is.null(baseline_df))
    geom_point(
      data  = baseline_df,
      aes(x = 0, y = level_base, colour = fill_group),
      size  = point_size, shape = 19, na.rm = TRUE,
      inherit.aes = FALSE
    )} +
  scale_y_discrete(
    labels = y_struct$labels,
    limits = rev(y_struct$levels),
    drop   = FALSE
  ) +
  scale_fill_manual(
    name     = NULL,
    values   = c(row1_values, row2_values),
    breaks   = c(row1_breaks, row2_breaks),
    labels   = c(row1_labels, row2_labels),
    na.value = NA,
    guide    = guide_legend(
      nrow         = 1,
      override.aes = list(
        fill   = c(attr_colours["attr_vicinity"], sp_light, attr_colours["attr_source_purpose"], unname(row2_values)),
        alpha  = c(0.7, 1.0, 0.7, rep(0.7, 4)),
        colour = NA,
        size   = 5
      )
    )
  ) +
  scale_colour_manual(values = c(row1_values, row2_values), na.value = NA, guide = "none") +
  facet_wrap(~ country_label, ncol = 2) +
  labs(x = "Preferences per country", y = NULL) +
  theme_classic(base_size = base_size) +
  theme(
    axis.text.y          = element_markdown(lineheight = 1.2),
    panel.grid.major.x   = element_line(color = "grey92", linewidth = 0.4),
    strip.text           = element_text(face = "bold", size = base_size),
    strip.background     = element_blank(),
    plot.margin          = margin(10, 20, 10, 10),
    legend.position      = "bottom",
    legend.justification = "center"
  )

ggsave(
  "output/plots/country_total.png",
  width = fig_width_w, height = 10, dpi = dpi_val, bg = "white"
)
