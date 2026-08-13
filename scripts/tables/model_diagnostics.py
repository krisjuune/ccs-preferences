"""
MCMC diagnostics table for the main hybrid choice model.

Outputs a LaTeX longtable with Mean, SD, ESS_bulk, ESS_tail, and R_hat for
all structural parameters: alpha (population-level), beta, gamma, delta,
beta_interact, gamma_interact, and theta_lreco / theta_galtan / theta_ecol.
Individual-level alpha and latent variables are excluded.

Run: python scripts/tables/model_diagnostics.py
Output: output/tables/model_diagnostics.tex
"""
import os
import re
import arviz as az

try:
    idata_path = snakemake.input[0]
    out_path   = snakemake.output[0]
except NameError:
    idata_path = "output/data/inference_main_hybrid_choice.nc"
    out_path   = "output/tables/model_diagnostics.tex"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

# ---- display name mappings ----

LEVEL_DISPLAY = {
    "attr_engagement_consult":                   "Engagement: Consult",
    "attr_engagement_vote":                      "Engagement: Vote",
    "attr_vicinity_another region":              "Proximity: Another region",
    "attr_vicinity_your region":                 "Proximity: Your region",
    "attr_vicinity_your municipality":           "Proximity: Your municipality",
    "attr_industry_metal and cement production": "Industry: Metal \\& cement",
    "attr_industry_gas with CCS":                "Industry: Gas with CCS",
    "attr_costs_polluting industry":             "Costs: Polluting industry",
    "attr_reason_close to source":               "Location reason: Close to source",
    "attr_reason_cost-efficient":                "Location reason: Cost-efficient",
    "attr_source_purpose_foreign":               "Source/purpose: Foreign",
    "attr_vicinity_another region::attr_source_purpose_foreign":  "Another region $\\times$ Foreign",
    "attr_vicinity_your region::attr_source_purpose_foreign":     "Your region $\\times$ Foreign",
    "attr_vicinity_your municipality::attr_source_purpose_foreign": "Your municipality $\\times$ Foreign",
}

# Preferred display order within parameter groups
LEVEL_ORDER = [
    "attr_engagement_consult",
    "attr_engagement_vote",
    "attr_vicinity_another region",
    "attr_vicinity_your region",
    "attr_vicinity_your municipality",
    "attr_industry_metal and cement production",
    "attr_industry_gas with CCS",
    "attr_costs_polluting industry",
    "attr_reason_close to source",
    "attr_reason_cost-efficient",
    "attr_source_purpose_foreign",
    "attr_vicinity_another region::attr_source_purpose_foreign",
    "attr_vicinity_your region::attr_source_purpose_foreign",
    "attr_vicinity_your municipality::attr_source_purpose_foreign",
]


def level_rank(level):
    try:
        return LEVEL_ORDER.index(level)
    except ValueError:
        return 999


# ---- load and summarise ----

print("Loading posterior...")
idata = az.from_netcdf(idata_path)

var_names = [
    "alpha_mu", "alpha_sigma",
    "beta", "delta", "beta_interact",
    "gamma", "gamma_interact",
    "theta_lreco", "theta_galtan", "theta_ecol",
]

print("Computing MCMC diagnostics...")
summary = az.summary(idata, var_names=var_names, round_to=None)
summary = summary[["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]]

# ---- index parsing ----

def parse_idx(idx):
    """Return (base, country_or_None, level_or_None) from az.summary index."""
    m = re.match(r'^([^\[]+)(?:\[(.+)\])?$', idx)
    base     = m.group(1)
    raw      = m.group(2)
    if raw is None:
        return base, None, None
    # 2D params: "country, level" — split on first ", " only
    if ", " in raw:
        country, level = raw.split(", ", 1)
        return base, country.strip(), level.strip()
    return base, None, raw.strip()


parsed = {idx: parse_idx(idx) for idx in summary.index}

# ---- formatters ----

def fmt(x, dec=3):
    return f"{x:.{dec}f}"

def fmt_ess(x):
    return f"{int(round(x)):,}"

def fmt_rhat(x):
    s = fmt(x)
    return f"\\textbf{{{s}}}" if x > 1.01 else s

def data_row(label, idx):
    r = summary.loc[idx]
    cells = [
        label,
        fmt(r["mean"]),
        fmt(r["sd"]),
        fmt_ess(r["ess_bulk"]),
        fmt_ess(r["ess_tail"]),
        fmt_rhat(r["r_hat"]),
    ]
    return " & ".join(cells) + " \\\\"

# ---- structured parameter groups ----

def rows_matching(base, country=None):
    """Return sorted list of (display_label, summary_index) for a parameter group."""
    out = []
    for idx, (b, c, level) in parsed.items():
        if b != base:
            continue
        if country is not None and c != country:
            continue
        if country is None and c is not None:
            continue  # skip country-specific rows when not requested
        label = LEVEL_DISPLAY.get(level, level) if level else idx
        out.append((label, idx, level or ""))
    out.sort(key=lambda x: level_rank(x[2]))
    return [(lbl, idx) for lbl, idx, _ in out]


GROUPS = [
    ("Positional bias ($\\alpha$)", [
        ("$\\alpha_\\mu$",    "alpha_mu"),
        ("$\\alpha_\\sigma$", "alpha_sigma"),
    ]),
    ("Main effects ($\\beta$)", rows_matching("beta")),
    ("CO\\textsubscript{2} framing ($\\delta$)", [
        ("Framing effect", "delta"),
    ]),
    ("Proximity $\\times$ framing interactions ($\\beta_{\\times}$)",
        rows_matching("beta_interact")),
    ("Country deviations ($\\gamma$) --- China",
        rows_matching("gamma", country="china")),
    ("Country deviations ($\\gamma$) --- Switzerland",
        rows_matching("gamma", country="switzerland")),
    ("Country $\\times$ proximity $\\times$ framing ($\\gamma_{\\times}$) --- China",
        rows_matching("gamma_interact", country="china")),
    ("Country $\\times$ proximity $\\times$ framing ($\\gamma_{\\times}$) --- Switzerland",
        rows_matching("gamma_interact", country="switzerland")),
    ("Value moderation, left-right economics ($\\theta_\\text{lreco}$)",
        rows_matching("theta_lreco")),
    ("Value moderation, GAL-TAN ($\\theta_\\text{galtan}$)",
        rows_matching("theta_galtan")),
    ("Value moderation, socio-ecological ($\\theta_\\text{ecol}$)",
        rows_matching("theta_ecol")),
]

# ---- build LaTeX ----

L = []

L.append("% Generated by scripts/tables/model_diagnostics.py")
L.append("% Requires: booktabs, longtable")
L.append("")
L.append("\\begin{longtable}{lrrrrr}")
L.append("\\caption{MCMC diagnostics for the main hybrid choice model. "
         "ESS: effective sample size. $\\hat{R}$: potential scale reduction factor "
         "(values $>1.01$ in bold).}")
L.append("\\label{tab:model_diagnostics}")
L.append("\\\\")
L.append("\\toprule")
L.append("Parameter & Mean & SD & "
         "ESS$_\\text{bulk}$ & ESS$_\\text{tail}$ & $\\hat{R}$ \\\\")
L.append("\\midrule")
L.append("\\endfirsthead")
L.append("")
L.append("\\multicolumn{6}{l}{\\textit{\\small Table~\\thetable\\ continued}} \\\\[2pt]")
L.append("\\toprule")
L.append("Parameter & Mean & SD & "
         "ESS$_\\text{bulk}$ & ESS$_\\text{tail}$ & $\\hat{R}$ \\\\")
L.append("\\midrule")
L.append("\\endhead")
L.append("")
L.append("\\midrule")
L.append("\\multicolumn{6}{r}{\\textit{\\small Continued on next page}} \\\\")
L.append("\\endfoot")
L.append("")
L.append("\\bottomrule")
L.append("\\endlastfoot")

first_group = True
for title, entries in GROUPS:
    if not entries:
        continue
    if not first_group:
        L.append("\\addlinespace[6pt]")
    first_group = False
    L.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}} \\\\")
    L.append("\\addlinespace[2pt]")
    for label, idx in entries:
        L.append("\\quad " + data_row(label, idx))

L.append("")
L.append("\\end{longtable}")

output = "\n".join(L)
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"Saved: {out_path}")
