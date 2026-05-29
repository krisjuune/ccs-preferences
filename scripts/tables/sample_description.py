import os
import numpy as np
import pandas as pd

# ---- load data (one row per respondent) ----

ch = pd.read_csv("data/data_translated_ch.csv", low_memory=False).drop_duplicates("id")
cn = pd.read_csv("data/data_translated_cn.csv", low_memory=False).drop_duplicates("id")

# unified education column name
ch["education"] = ch["education_degree"]

# CN age: aggregate 55-64 and 65 or older into one bin (matches quota design)
cn["age"] = cn["age"].replace("65 or older", "55-64")

# qualitative income labels — both countries have 6 ordered brackets
ch_income_map = {
    "Under CHF 34,000":        "Lowest (${<}$34k CHF / ${<}$10k \\textyen)",
    "CHF 34,001 – CHF 46,000": "Second (34--46k CHF / 10--20k \\textyen)",
    "CHF 45,001 – CHF 59,000": "Third (45--59k CHF / 20--30k \\textyen)",
    "CHF 59,001 – CHF 77,000": "Fourth (59--77k CHF / 30--50k \\textyen)",
    "CHF 77,001 – CHF 96,000": "Fifth (77--96k CHF / 50--100k \\textyen)",
    "Above CHF 96,000":        "Highest (${>}$96k CHF / ${>}$100k \\textyen)",
    "Prefer not to say":       "Prefer not to say",
}
cn_income_map = {
    "10,000元以下":      "Lowest (${<}$34k CHF / ${<}$10k \\textyen)",
    "10,001–20,000元 ":  "Second (34--46k CHF / 10--20k \\textyen)",
    "20,001–30,000元 ":  "Third (45--59k CHF / 20--30k \\textyen)",
    "30,001–50,000元 ":  "Fourth (59--77k CHF / 30--50k \\textyen)",
    "50,001–100,000元 ": "Fifth (77--96k CHF / 50--100k \\textyen)",
    "100,000 元以上 ":   "Highest (${>}$96k CHF / ${>}$100k \\textyen)",
    "不愿透露":           "Prefer not to say",
}
ch["income_q"] = ch["income"].map(ch_income_map).fillna(ch["income"])
cn["income_q"] = cn["income"].map(cn_income_map).fillna(cn["income"])

# ---- target quotas (from scripts/quota_check.py) ----

targets = {
    "CH": {
        "age": {
            "18-24": 0.09, "25-34": 0.16, "35-44": 0.18,
            "45-54": 0.17, "55-64": 0.17, "65 or older": 0.23,
        },
        "gender": {"Male": 0.50, "Female": 0.50},
        "language_region": {
            "German-speaking region": 0.72,
            "French-speaking region": 0.28,
        },
    },
    "CN": {
        "age": {
            "18-24": 0.12, "25-34": 0.20, "35-44": 0.18,
            "45-54": 0.16, "55-64": 0.34,
        },
        "gender": {"Male": 0.51, "Female": 0.49},
    },
}

# ---- helpers ----

def pct(val):
    return f"{val * 100:.1f}\\%" if not pd.isna(val) else ""

def target(country, var, cat):
    val = targets.get(country, {}).get(var, {}).get(cat)
    return pct(val) if val is not None else ""

def props(df, col, cats):
    base = df[col].dropna()
    n = len(base)
    return {c: (base == c).sum() / n if n > 0 else np.nan for c in cats}

def row(label, ch_s, ch_t, cn_s, cn_t, indent=True):
    pfx = "\\quad " if indent else ""
    return f"{pfx}{label} & {pct(ch_s)} & {ch_t} & {pct(cn_s)} & {cn_t} \\\\\n"

# ---- build table body ----

lines = []

# sample size
lines += ["\\addlinespace\n",
    f"\\textbf{{Sample size}} & $n$ = {len(ch)} & & $n$ = {len(cn)} & \\\\\n"]

# age
lines += ["\\addlinespace\n", "\\textbf{Age} \\\\\n"]
age_ch = props(ch, "age", ["18-24","25-34","35-44","45-54","55-64","65 or older"])
age_cn = props(cn, "age", ["18-24","25-34","35-44","45-54","55-64"])
for cat, label in [("18-24","18--24"),("25-34","25--34"),("35-44","35--44"),
                   ("45-54","45--54"),("55-64","55--64"),("65 or older","65+")]:
    lines.append(row(label,
        age_ch.get(cat, np.nan), target("CH","age",cat),
        age_cn.get(cat, np.nan), target("CN","age",cat)))

# gender
lines += ["\\addlinespace\n", "\\textbf{Gender} \\\\\n"]
for cat in ["Male","Female","Non-binary / third gender"]:
    lines.append(row(cat,
        props(ch,"gender",[cat]).get(cat, np.nan), target("CH","gender",cat),
        props(cn,"gender",[cat]).get(cat, np.nan), target("CN","gender",cat)))

# language region (CH only)
lines += ["\\addlinespace\n", "\\textbf{Language region (CH)} \\\\\n"]
reg_cats = ["German-speaking region","French-speaking region",
            "Italian-speaking region","Romansh-speaking region"]
reg_short = {"German-speaking region":"German-speaking",
             "French-speaking region":"French-speaking",
             "Italian-speaking region":"Italian-speaking",
             "Romansh-speaking region":"Romansh-speaking"}
ch_reg = props(ch, "language_region", reg_cats)
for cat in reg_cats:
    lines.append(row(reg_short[cat],
        ch_reg.get(cat, np.nan), target("CH","language_region",cat),
        np.nan, ""))

# education (both countries)
lines += ["\\addlinespace\n", "\\textbf{Education} \\\\\n"]
edu_map = {
    "No high school diploma":                                "No high school diploma",
    "High school or vocational training":                    "High school / vocational",
    "Bachelor":                                              "Bachelor's",
    "Master":                                                "Master's",
    "Doctoral or professional degree (e.g., PhD, MD, JD)":  "Doctoral / professional",
    "Other (please specify)":                                "Other",
    "Prefer not to say":                                     "Prefer not to say",
}
ch_edu = props(ch, "education", list(edu_map.keys()))
cn_edu = props(cn, "education", list(edu_map.keys()))
for cat, label in edu_map.items():
    lines.append(row(label,
        ch_edu.get(cat, np.nan), "",
        cn_edu.get(cat, np.nan), ""))

# income (shared qualitative labels)
lines += ["\\addlinespace\n", "\\textbf{Income} \\\\\n"]
income_cats = list(dict.fromkeys(ch_income_map.values()))  # preserves insertion order
ch_inc = props(ch, "income_q", income_cats)
cn_inc = props(cn, "income_q", income_cats)
for cat in income_cats:
    lines.append(row(cat,
        ch_inc.get(cat, np.nan), "",
        cn_inc.get(cat, np.nan), ""))

# living area
lines += ["\\addlinespace\n", "\\textbf{Living area} \\\\\n"]
area_map = {
    "Big city":                                      "Big city",
    "Outer neighborhood or suburb of a large city":  "Suburb of large city",
    "Medium-sized or small town":                    "Medium / small town",
    "Village":                                       "Village",
    "Farm":                                          "Farm",
    "Other":                                         "Other",
}
ch_area = props(ch, "living_area", list(area_map.keys()))
cn_area = props(cn, "living_area", list(area_map.keys()))
for cat, label in area_map.items():
    lines.append(row(label,
        ch_area.get(cat, np.nan), "",
        cn_area.get(cat, np.nan), ""))

# survey duration
lines += ["\\addlinespace\n"]
ch_dur = ch["Duration (in seconds)"].median() / 60
cn_dur = cn["Duration (in seconds)"].median() / 60
lines.append(
    f"\\textbf{{Median survey duration}} & {ch_dur:.1f} min & & {cn_dur:.1f} min & \\\\\n"
)

# ---- assemble ----

header = (
    "\\begin{table}[H]\n"
    "\\centering\n"
    "\\small\n"
    "\\begin{tabular}{lrrrr}\n"
    "\\toprule\n"
    " & \\multicolumn{2}{c}{Switzerland} & \\multicolumn{2}{c}{China} \\\\\n"
    "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
    "\\textbf{Variable} & Sample & Target & Sample & Target \\\\\n"
    "\\midrule\n"
)

footer = (
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\caption{Sample description. Target quotas for age and gender follow national "
    "census distributions; language region targets follow Swiss Federal Statistical "
    "Office figures. Income brackets differ by country and are mapped to shared "
    "ordinal labels for comparability.}\n"
    "\\label{tab:sample}\n"
    "\\end{table}\n"
)

os.makedirs("output/tables", exist_ok=True)
with open("output/tables/sample_description.tex", "w") as f:
    f.write(header + "".join(lines) + footer)

print("Saved to output/tables/sample_description.tex")
