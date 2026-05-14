library(tidyverse)
library(ggdist)
library(ggtext)

# ---- load data ----

beta <- read_csv("output/data/posteriors_beta.csv")

# ---- attribute ordering and display ----

attr_order <- c(
  "attr_engagement",
  "attr_vicinity",
  "attr_industry",
  "attr_costs",
  "attr_reason",
  "attr_source_purpose"
)

attr_display <- c(
  "attr_engagement"     = "Engagement",
  "attr_vicinity"       = "Vicinity",
  "attr_industry"       = "Industry",
  "attr_costs"          = "Costs",
  "attr_reason"         = "Location reason",
  "attr_source_purpose" = "Source / Purpose"
)

attr_colours <- c(
  "attr_engagement"     = "#4e79a7",
  "attr_vicinity"       = "#f28e2b",
  "attr_industry"       = "#e15759",
  "attr_costs"          = "#76b7b2",
  "attr_reason"         = "#59a14f",
  "attr_source_purpose" = "#b07aa1"
)

# ---- level labels (keyed on base level name, no framing suffix) ----

level_display <- c(
  "attr_engagement_inform"                     = "Inform",
  "attr_engagement_vote"                       = "Vote",
  "attr_engagement_consult"                    = "Consult",
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

# ---- prepare data ----

plot_data <- beta |>
  filter(model == "hybrid") |>
  filter(
    # non-source_purpose: use average (framing-pooled)
    (framing == "average" & !grepl("source_purpose", level)) |
    # source_purpose: use framing-specific rows
    (framing %in% c("source", "purpose") & grepl("source_purpose", level))
  ) |>
  mutate(
    # strip framing suffix so source and purpose share a y-position
    level_base = gsub("_(source|purpose)$", "", level),
    attribute  = sapply(level_base, get_attribute)
  ) |>
  filter(level_base %in% names(level_display)) |>
  mutate(level_base = factor(level_base, levels = y_struct$levels))

# empty header rows (no distribution plotted, just y-axis label)
header_rows <- tibble(
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
)

plot_data <- bind_rows(plot_data, header_rows) |>
  mutate(
    level_base     = factor(level_base, levels = y_struct$levels),
    slab_alpha_val = if_else(framing == "purpose", 0.18, 0.7)  # kept for reference
  )

# ---- plot ----

ggplot(
  plot_data,
  aes(
    x      = value,
    y      = level_base,
    group  = interaction(level_base, framing),
    fill   = attribute,
    colour = attribute
  )
) +
  stat_halfeye(
    data           = \(d) filter(d, framing != "purpose"),
    slab_alpha     = 0.6,
    point_size     = 3,
    .width         = 0.9,
    point_interval = median_hdi,
    slab_colour    = NA,
    na.rm          = TRUE
  ) +
  stat_halfeye(
    data           = \(d) filter(d, framing == "purpose"),
    slab_alpha     = 0.3,
    point_alpha    = 0.6,
    interval_alpha = 0.6,
    point_size     = 3,
    .width         = 0.9,
    point_interval = median_hdi,
    slab_colour    = NA,
    na.rm          = TRUE
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.5) +
  scale_y_discrete(
    labels = y_struct$labels,
    limits = rev(y_struct$levels),
    drop   = FALSE
  ) +
  scale_fill_manual(values = attr_colours, na.value = NA, guide = "none") +
  scale_colour_manual(values = attr_colours, na.value = NA, guide = "none") +
  labs(x = "Partworth utility", y = NULL) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.y        = element_markdown(lineheight = 1.2),
    panel.grid.major.x = element_line(color = "grey92", linewidth = 0.4),
    plot.margin        = margin(10, 20, 10, 10)
  )

ggsave("output/plots/partworths.png", width = 10, height = 10, dpi = 300, bg = "white")
