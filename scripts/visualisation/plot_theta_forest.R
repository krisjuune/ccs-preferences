library(tidyverse)
library(ggdist)
library(ggtext)

# ---- load data ----

theta <- read_csv("output/data/posteriors_theta.csv")

# ---- attribute setup (shared with other plot scripts) ----

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

dim_order <- c("lreco", "galtan", "ecol")

dim_labels <- c(
  "lreco"  = "Left-right economic",
  "galtan" = "GAL-TAN",
  "ecol"   = "Socio-ecological"
)

# ---- prepare data ----

plot_data <- theta |>
  filter(model == "hybrid") |>
  filter(level %in% names(level_display)) |>
  mutate(
    level_base = factor(level, levels = y_struct$levels),
    attribute  = sapply(level, get_attribute),
    dim        = factor(dim, levels = dim_order),
    dim_label  = dim_labels[dim]
  )

header_rows <- crossing(
  tibble(
    level_base = factor(
      grep("^header_", y_struct$levels, value = TRUE),
      levels = y_struct$levels
    ),
    value     = NA_real_,
    model     = "hybrid",
    chain     = 1L,
    draw      = 1L,
    attribute = NA_character_
  ),
  dim       = factor(dim_order, levels = dim_order),
  dim_label = unname(dim_labels)
)

plot_data <- bind_rows(plot_data, header_rows) |>
  mutate(level_base = factor(level_base, levels = y_struct$levels))

# ---- plot ----

ggplot(plot_data, aes(x = value, y = level_base, fill = attribute, colour = attribute)) +
  stat_halfeye(
    slab_alpha     = 0.6,
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
  facet_wrap(~ dim_label, ncol = 3) +
  labs(x = "Value moderation effect (θ)", y = NULL) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.y        = element_markdown(lineheight = 1.2),
    panel.grid.major.x = element_line(color = "grey92", linewidth = 0.4),
    strip.text         = element_text(face = "bold", size = 14),
    strip.background   = element_blank(),
    plot.margin        = margin(10, 20, 10, 10)
  )

ggsave(
  "output/plots/theta_forest.png",
  width = 20, height = 10, dpi = 300, bg = "white"
)
