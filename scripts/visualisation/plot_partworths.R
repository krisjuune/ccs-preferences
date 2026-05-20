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
dpi_val      <- plt$dpi
sp_light     <- plt$sp_source_light

# baseline level names (one per attribute, fixed to 0 under reference_level coding)
baseline_level_names <- c(
  "attr_engagement_inform",
  "attr_vicinity_abroad",
  "attr_industry_waste incineration",
  "attr_costs_taxpayer",
  "attr_reason_sparsely-populated",
  "attr_source_purpose_domestic"
)

# ---- load data ----

beta <- read_csv("output/data/posteriors_beta.csv")

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
  "attr_vicinity"       = "Vicinity",
  "attr_industry"       = "Industry",
  "attr_costs"          = "Cost responsibility",
  "attr_reason"         = "Location reason",
  "attr_source_purpose" = "Source / Purpose"
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
    level_base = factor(level_base, levels = y_struct$levels),
    fill_group = case_when(
      is.na(attribute)  ~ NA_character_,
      framing == "purpose" & grepl("source_purpose", as.character(level_base)) ~ "sp_purpose",
      grepl("source_purpose", as.character(level_base)) ~ "sp_source",
      TRUE ~ attribute
    )
  )

# Row 1 legend (fill): vicinity + two purples — independent column widths
row1_breaks <- c("attr_vicinity", "sp_source", "sp_purpose")
row1_values <- c(
  "attr_vicinity" = "#f28e2b",
  "sp_source"     = "#b07aa1",
  "sp_purpose"    = "#b07aa1"   # same for rendering; legend key lightened via override.aes
)
row1_labels <- c(
  "attr_vicinity" = "Vicinity",
  "sp_source"     = "Source / Purpose (source framing)",
  "sp_purpose"    = "Source / Purpose (purpose framing)"
)

# Row 2 legend (colour): remaining four attributes
row2_breaks <- c("attr_industry", "attr_costs", "attr_reason", "attr_engagement")
row2_values <- attr_colours[row2_breaks]
row2_labels <- c(
  "attr_industry"   = "Industry",
  "attr_costs"      = "Cost responsibility",
  "attr_reason"     = "Location reason",
  "attr_engagement" = "Engagement"
)

# baseline rows (reference_level only): implicit value = 0, shown as a dot
baseline_df <- if (coding == "reference_level") {
  tibble(
    level_base = factor(baseline_level_names, levels = y_struct$levels),
    fill_group = if_else(
      grepl("source_purpose", baseline_level_names),
      "sp_source",
      sapply(baseline_level_names, get_attribute)
    )
  )
} else NULL

# ---- plot ----

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
  # row 1: vicinity + two purple framing entries (sized by their own labels)
  scale_fill_manual(
    name     = NULL,
    values   = c(row1_values, row2_values),
    breaks   = row1_breaks,
    labels   = row1_labels,
    na.value = NA,
    guide    = guide_legend(
      nrow         = 1,
      override.aes = list(
        fill   = c(attr_colours["attr_vicinity"], sp_light, attr_colours["attr_source_purpose"]),
        alpha  = c(0.7, 1.0, 0.7),
        colour = NA,
        size   = 5
      ),
      order = 1
    )
  ) +
  # row 2: remaining four attributes (sized by their own shorter labels)
  scale_colour_manual(
    name     = NULL,
    values   = c(row1_values, row2_values),
    breaks   = row2_breaks,
    labels   = row2_labels,
    na.value = NA,
    guide    = guide_legend(
      nrow         = 1,
      override.aes = list(
        fill   = unname(row2_values),
        colour = NA,
        alpha  = rep(0.7, 4),
        size   = 5
      ),
      order = 2
    )
  ) +
  labs(x = "Partworth utility", y = NULL) +
  theme_classic(base_size = base_size) +
  theme(
    axis.text.y          = element_markdown(lineheight = 1.2),
    panel.grid.major.x   = element_line(color = "grey92", linewidth = 0.4),
    plot.margin          = margin(10, 20, 10, 10),
    legend.position      = "bottom",
    legend.justification = "center",
    legend.box           = "vertical",
    legend.box.just      = "left"
  )

ggsave("output/plots/partworths.png", width = 10, height = 10, dpi = dpi_val, bg = "white")
