library(tidyverse)
library(ggdist)
library(yaml)

# ---- settings ----

config        <- read_yaml("config.yaml")
plt           <- config$plots
value_colours <- unlist(plt$value_colours)
base_size     <- plt$base_size
fig_width_w   <- plt$fig_width_wide
dpi_val       <- plt$dpi
ci_width      <- plt$ci_width
point_size    <- plt$point_size

# ---- load data ----

loadings <- read_csv("output/data/posteriors_loadings.csv")

# ---- dimension and item labels ----

dim_order <- c("lreco", "galtan", "ecol")
dim_labels <- c(
  "lreco"  = "Left-right economic",
  "galtan" = "GAL-TAN",
  "ecol"   = "Socio-ecological"
)

item_order <- c("item_1", "item_2", "item_3")
item_display <- c(
  "item_1" = "Item 1 (fixed = 1)",
  "item_2" = "Item 2",
  "item_3" = "Item 3"
)

# dim_colours keyed on the loadings data's "ecol" (value_colours uses "socio_ecol")
dim_colours <- c(
  "lreco"  = value_colours[["lreco"]],
  "galtan" = value_colours[["galtan"]],
  "ecol"   = value_colours[["socio_ecol"]]
)

# ---- prepare data ----

plot_data <- loadings |>
  filter(model == "hybrid") |>
  mutate(
    dim_label  = factor(dim_labels[dim], levels = unname(dim_labels)),
    item_label = factor(item_display[item], levels = unname(item_display))
  )

# item 1 per dimension is fixed to 1 (not estimated) — shown as a reference dot
fixed_df <- crossing(
  tibble(dim = dim_order),
  tibble(item = "item_1")
) |>
  mutate(
    dim_label  = factor(dim_labels[dim], levels = unname(dim_labels)),
    item_label = factor(item_display[item], levels = unname(item_display))
  )

# ---- plot ----

ggplot(plot_data, aes(x = value, y = item_label, colour = dim)) +
  stat_pointinterval(
    point_size     = point_size,
    .width         = ci_width,
    point_interval = median_hdi,
    na.rm          = TRUE
  ) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
  geom_point(
    data  = fixed_df,
    aes(x = 1, y = item_label, colour = dim),
    size  = point_size, shape = 19, na.rm = TRUE,
    inherit.aes = FALSE
  ) +
  scale_y_discrete(
    limits = rev(item_order |> (\(x) item_display[x])() |> unname()),
    drop   = FALSE
  ) +
  scale_colour_manual(values = dim_colours, guide = "none") +
  facet_wrap(~ dim_label, ncol = 3) +
  labs(x = "Factor loading (λ)", y = NULL) +
  theme_classic(base_size = base_size) +
  theme(
    strip.text         = element_text(face = "bold", size = base_size),
    strip.background   = element_blank(),
    panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.4),
    plot.margin        = margin(10, 20, 10, 10)
  )

ggsave(
  "output/plots/supp_figs/factor_loadings.png",
  width = fig_width_w, height = 3, dpi = dpi_val, bg = "white"
)
