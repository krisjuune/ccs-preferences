library(tidyverse)
library(ggdist)
library(yaml)

config       <- read_yaml("config.yaml")
plt          <- config$plots
attr_colours <- unlist(plt$attr_colours)
base_size    <- plt$base_size
dpi_val      <- plt$dpi
slab_alpha   <- plt$slab_alpha_framing
point_size   <- plt$point_size
ci_width     <- plt$ci_width
fig_width    <- plt$fig_width

coefs <- read_csv("output/data/posteriors_interact_coefs.csv")

# ---- clean labels: "Another region × Foreign" from raw column names ----

attr_re <- paste(
  "attr_engagement_", "attr_vicinity_", "attr_industry_",
  "attr_costs_", "attr_reason_", "attr_source_purpose_",
  sep = "|"
)

clean_level_name <- function(x) {
  str_remove(x, attr_re) |>
    str_to_sentence() |>
    str_replace("\\bccs\\b", "CCS")
}

clean_interact_label <- function(x) {
  sapply(strsplit(x, "::"), function(parts) {
    paste(clean_level_name(parts[1]), clean_level_name(parts[2]), sep = " × ")
  })
}

# ---- colour by the second attribute in the interaction pair ----

beta_df <- coefs |>
  filter(param == "beta_interact") |>
  mutate(
    interact_label = clean_interact_label(interact),
    fill_colour    = case_when(
      str_detect(interact, "attr_vicinity") ~ attr_colours["attr_vicinity"],
      str_detect(interact, "attr_costs")   ~ attr_colours["attr_costs"],
      TRUE ~ "grey50"
    )
  )

# explicit ordering: proximity × source, then proximity × close, then costs × industry
level_order <- c(
  "Another region × Foreign",
  "Your region × Foreign",
  "Your municipality × Foreign",
  "Another region × Close to source",
  "Your region × Close to source",
  "Your municipality × Close to source",
  "Polluting industry × Metal and cement production",
  "Polluting industry × Gas with CCS"
)

beta_df <- beta_df |>
  mutate(interact_label = factor(interact_label, levels = rev(level_order)))

# ---- plot ----

ggplot(beta_df, aes(x = value, y = interact_label,
                    fill = fill_colour, colour = fill_colour)) +
  stat_halfeye(
    slab_alpha     = slab_alpha,
    point_size     = point_size,
    .width         = ci_width,
    point_interval = median_hdi,
    slab_colour    = NA,
    na.rm          = TRUE
  ) +
  geom_vline(xintercept = 0, linetype = "dashed",
             colour = "grey40", linewidth = 0.5) +
  scale_fill_identity() +
  scale_colour_identity() +
  labs(x = "Interaction effect", y = NULL) +
  theme_classic(base_size = base_size) +
  theme(
    panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.4),
    plot.margin        = margin(10, 20, 10, 10)
  )

ggsave("output/plots/supp_figs/interact_forest_beta.png",
       width = fig_width, height = 6, dpi = dpi_val, bg = "white")
