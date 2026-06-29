# CCS siting preferences — choice experiment in Switzerland and China

Snakemake pipeline for a discrete choice experiment on public preferences
for carbon capture and storage (CCS) site characteristics in Switzerland and
China. The pipeline preprocesses raw survey exports, fits a Bayesian hybrid
choice model (HCM) in PyMC, and produces postprocessed tables and figures.

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate ccs-location
```

`environment.yml` installs Python (PyMC, ArviZ, pandas, etc.) and R
(tidyverse, ggdist, ggtext, yaml) dependencies via conda, and Snakemake via
`pip` (the conda package does not reliably install on all platforms).

Raw survey exports go in `raw_data/`; the files used are set in
`config.yaml` under `raw_data:`.

## Running the pipeline

Run everything:

```bash
snakemake --cores 4
```

Run a specific output (and everything it depends on):

```bash
snakemake --cores 4 output/plots/supp_figs/partworths.png
```

Force-rerun a specific rule (e.g. after editing a plotting script, when
Snakemake thinks the output is already up to date):

```bash
snakemake --cores 4 -R plot_partworths
```

## Pipeline stages

1. **Preprocessing** (`scripts/preprocessing/`) — clean raw exports, build
   the three latent value indices from Likert items (`value_indices.py`),
   reshape the conjoint tasks to long format and assign framing conditions
   (`translate_conjoints.py`).
2. **Choice models** (`scripts/analysis/`) — fit in PyMC:
   - `basic_choice_model.py` — simple multinomial choice model (no latent
     traits), off by default (`run_basic_model: false`)
   - `hybrid_choice_model.py` — the main model: measurement model (CFA-style
     factor loadings) for the latent value dimensions, plus a choice model
     where preferences are moderated by those latent traits
   - `interaction_choice_model.py` — adds cherry-picked attribute
     interactions, incl. a three-way country × proximity × framing term
     (enabled via `run_interaction_model: true`)
   - `full_interaction_choice_model.py` — all pairwise attribute-level
     interactions (enabled via `run_full_interaction_model: true`)
3. **Postprocessing** (`scripts/postprocessing/`) — extract posterior
   samples into tidy CSVs for plotting (partworths, country-specific
   utilities, value moderation effects, factor loadings, interaction
   effects).
4. **Visualisation** (`scripts/visualisation/`) — R/ggplot2 scripts, one per
   figure, reading the postprocessed CSVs.
5. **Tables** (`scripts/tables/`) and **value correlations**
   (`scripts/analysis/value_correlations.py`) — descriptive sample
   statistics and value/demographic correlation matrices.

## Configuration (`config.yaml`)

- `raw_data` — paths to the raw CH/CN survey export files.
- `run_basic_model` / `run_interaction_model` / `run_full_interaction_model`
  — toggle optional model variants on/off.
- `coding` — `"sum_to_zero"` or `"reference_level"`, controls how attribute
  levels are dummy-coded across all choice models (see
  `hybrid_choice_model.py` for the baseline level chosen per attribute).
- `mcmc` — sampler settings (`seed`, `draws`, `tune`, `chains`, `cores`),
  shared by all choice models.
- `plots` — shared visual settings (sizes, colours, alphas) read by every
  R script in `scripts/visualisation/`, so figure styling stays consistent
  without editing individual scripts.

## Outputs

- `output/data/` — posterior inference objects (`.nc`) and postprocessed
  posterior CSVs.
- `output/plots/` — figures. Core results live directly under
  `output/plots/`; diagnostic/supplementary figures (factor loadings, theta
  heatmap, general partworths, country-direct effect, interaction forest plot) are
  under `output/plots/supp_figs/`.
- `output/tables/` — LaTeX/CSV tables (sample description, value
  correlations).
