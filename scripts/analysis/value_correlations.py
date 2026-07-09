"""
Pairwise correlations between the three latent value indices (lreco, galtan,
socio_ecol) and basic socio-economic variables (age, education, income,
gender), computed separately for Switzerland and China.

Ordinal variables (age, education, income) are rank-coded; gender is coded
as a 0/1 indicator (female = 1). Correlations use pairwise-complete
observations (each pair excludes only rows missing on either variable).
"""
import os
import pandas as pd

try:
    out_ch = snakemake.output.ch
    out_cn = snakemake.output.cn
except NameError:
    out_ch = "output/tables/value_correlations_ch.csv"
    out_cn = "output/tables/value_correlations_cn.csv"

# ---- recoding maps ----

age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65 or older"]
age_rank = {cat: i + 1 for i, cat in enumerate(age_order)}

education_rank = {
    "No high school diploma": 1,
    "High school or vocational training": 2,
    "Bachelor": 3,
    "Master": 4,
    "Doctoral or professional degree (e.g., PhD, MD, JD)": 5,
    # "Other (please specify)" and "Prefer not to say" map to NaN (excluded)
}

income_rank_ch = {
    "Under CHF 34,000": 1,
    "CHF 34,001 – CHF 46,000": 2,
    "CHF 45,001 – CHF 59,000": 3,
    "CHF 59,001 – CHF 77,000": 4,
    "CHF 77,001 – CHF 96,000": 5,
    "Above CHF 96,000": 6,
    # "Prefer not to say" maps to NaN
}
income_rank_cn = {
    "10,000元以下": 1,
    "10,001–20,000元 ": 2,
    "20,001–30,000元 ": 3,
    "30,001–50,000元 ": 4,
    "50,001–100,000元 ": 5,
    "100,000 元以上 ": 6,
    # "不愿透露" maps to NaN
}


def load_demographics(path, country, income_rank):
    df = pd.read_csv(path, low_memory=False).drop_duplicates("id")
    # CH names the column "education_degree"; CN already calls it "education"
    education_col = "education" if "education" in df.columns else "education_degree"
    df["country"]        = country
    df["age_rank"]       = df["age"].map(age_rank)
    df["education_rank"] = df[education_col].map(education_rank)
    df["income_rank"]    = df["income"].map(income_rank)
    df["female"]         = df["gender"].map({"Female": 1, "Male": 0})
    return df[["id", "country", "age_rank", "education_rank", "income_rank", "female"]]


# ---- load and merge ----

ch_demo = load_demographics("data/data_translated_ch.csv", "switzerland", income_rank_ch)
cn_demo = load_demographics("data/data_translated_cn.csv", "china", income_rank_cn)
demo = pd.concat([ch_demo, cn_demo], ignore_index=True)

values = (
    pd.read_csv("data/data_values_ch_cn.csv")[["id", "country", "lreco", "galtan", "socio_ecol"]]
    .drop_duplicates()
)

data = values.merge(demo, on=["id", "country"], how="inner")

# ---- correlation matrices, per country ----

display_names = {
    "lreco":          "Socio-economic",
    "galtan":         "Socio-cultural",
    "socio_ecol":     "Socio-ecological",
    "age_rank":       "Age",
    "education_rank": "Education",
    "income_rank":    "Income",
    "female":         "Gender (female = 1)",
}
vars_of_interest = list(display_names.keys())

os.makedirs(os.path.dirname(out_ch), exist_ok=True)

for country, out_path in [("switzerland", out_ch), ("china", out_cn)]:
    sub = data.loc[data["country"] == country, vars_of_interest].rename(columns=display_names)
    corr = sub.corr(method="pearson")
    corr.to_csv(out_path)
    print(f"\n{country.title()} (n = {len(sub)} respondents):")
    print(corr.round(2))
    print(f"Saved: {out_path}")
