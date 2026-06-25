import pandas as pd
import re

# %% 

ch_df = pd.read_csv("data/data_untranslated_ch.csv")
cn_df = pd.read_csv("data/data_untranslated_cn.csv")

# %% 
def apply_mapping(df, mapping_dict, column_pattern=None):
    """
    Apply a mapping to columns in the DataFrame based on a dictionary, with
    an optional string or list of strings column_pattern to filter column names.
    If None, all columns are considered.
    """
    
     # turn column_pattern into a list it already isn't
    if isinstance(column_pattern, str):
        column_patterns = [column_pattern]
    elif isinstance(column_pattern, list) and all(isinstance(pat, str) for pat in column_pattern):
        column_patterns = column_pattern
    elif column_pattern is None:
        column_patterns = []
    else:
        raise ValueError("column_pattern should be a string, list of strings, or None.")
    
    # identify columns to apply the mapping
    if column_patterns:
        columns_to_map = [col for col in df.columns if any(pat in col for pat in column_patterns)]
    else:
        columns_to_map = df.columns
    
    # apply mapping to the identified columns
    for column in columns_to_map:
        df[column] = df[column].replace(mapping_dict)
    
    return df

def reshape_conjoint_to_long(df, respondent_id_col = None, country = "CH"):
    """
    Reshape randomized conjoint data into long format with
    one row per respondent, per task, per package (p1/p2)
    Columns are: respondent ID (optional), task, package, attr_[name]

    Parameters:
    df : pd.DataFrame wide-format dataframe
    respondent_id_col : str or None for respondent identifiers
    """
    n_rows = df.shape[0]
    name_cols = [col for col in df.columns if re.match(r"c\d+_atr\d+_name", col)]
    long_data = []

    for name_col in name_cols:
        match = re.match(r"c(\d+)_atr(\d+)_name", name_col)
        if not match:
            continue
        task, atr = match.groups()
        task = int(task)

        p1_col = f"c{task}_atr{atr}_p1"
        p2_col = f"c{task}_atr{atr}_p2"
        if p1_col not in df.columns or p2_col not in df.columns:
            continue

        choose_col = f"{task}_conjoint_choose12"
        plan1_col = f"{task}_conjoint_plan1"
        plan2_col = f"{task}_conjoint_plan2"

        for i in range(n_rows):
            attr = df[name_col].iloc[i]
            if pd.isna(attr):
                continue

            chosen_plan = df[choose_col].iloc[i] if choose_col in df.columns else None
            plan1_support = df[plan1_col].iloc[i] if plan1_col in df.columns else None
            plan2_support = df[plan2_col].iloc[i] if plan2_col in df.columns else None

            if pd.isna(chosen_plan):
                chosen_plan = None
            if isinstance(plan1_support, str):
                plan1_support = 1 if plan1_support == "In favor" else 0
            if isinstance(plan2_support, str):
                plan2_support = 1 if plan2_support == "In favor" else 0

            row_base = {
                "task": task,
                "chosen_plan": chosen_plan
            }
            if respondent_id_col:
                row_base["id"] = df[respondent_id_col].iloc[i]

            # Package 1
            long_data.append({
                **row_base,
                "package": "1",
                f"attr_{attr}": df[p1_col].iloc[i],
                "chosen": 1 if chosen_plan == "Plan 1" else 0,
                "supported": plan1_support
            })

            # Package 2
            long_data.append({
                **row_base,
                "package": "2",
                f"attr_{attr}": df[p2_col].iloc[i],
                "chosen": 1 if chosen_plan == "Plan 2" else 0,
                "supported": plan2_support
            })

    long_df = pd.DataFrame(long_data)

    if "attr_source" in long_df.columns or "attr_purpose" in long_df.columns:
        long_df["attr_source_purpose"] = long_df["attr_source"].combine_first(long_df["attr_purpose"])
        long_df["framing"] = None
        long_df.loc[long_df["attr_source"].notna(), "framing"] = "purpose"
        long_df.loc[long_df["attr_purpose"].notna(), "framing"] = "source"

        long_df = long_df.drop(columns=["attr_source", "attr_purpose"])

    if long_df.empty:
        print("Warning: No rows created. Check for missing attribute columns.")
        return long_df

    index_cols = ["task", "package"]
    if respondent_id_col:
        index_cols.insert(0, "id")

    long_df = long_df.groupby(["id", "task", "package"], as_index=False).first()

    attr_cols = [col for col in long_df.columns if col.startswith("attr_")]
    other_cols = [col for col in long_df.columns if not col.startswith("attr_")]
    long_df = long_df[other_cols + attr_cols]

    return long_df

# %% translate attribute names

attr_names_dict = {
    "Das Kohlendioxid wird gespeichert in": "vicinity",
    "The carbon dioxide is stored in": "vicinity",
    "Le dioxyde de carbone est stocké dans": "vicinity",
    "二氧化碳被储存在": "vicinity",

    "Dieser Standort wurde gewählt, weil er": "reason",
    "Cet endroit a été choisi parce qu’il": "reason",
    "This location was chosen because it": "reason",
    "选择储存在这里是因为": "reason",

    "CCS wird angewendet bei": "industry",
    "CCS is applied to": "industry",
    "Le CSC est appliqué aux": "industry",
    "碳捕集与封存被应用在": "industry",

    "Das gespeicherte Kohlendioxid stammt aus": "source",
    "The carbon dioxide stored will come from": "source",
    "Le dioxyde de carbone stocké proviendra": "source",
    "储存的二氧化碳来自": "source",

    "Der Zweck des Kohlendioxid-Speicherprojekts ist": "purpose",
    "The purpose of carbon dioxide storage project is": "purpose",
    "L’objectif du projet de stockage du dioxyde de carbone est": "purpose",
    "储存二氧化碳的目的是": "purpose",

    "Die Kosten für die Lagerung trägt": "costs",
    "The cost of storage will be borne by": "costs",
    "Les frais de stockage seront à la charge de": "costs",
    "储存费用由谁来承担": "costs",

    "Bei Entscheidungen über Speicherprojekte werden Sie": "engagement",
    "In decisions about storage projects, you will": "engagement",
    "Dans les décisions concernant les projets de stockage, vous allez": "engagement",
    "在储存项目的决策中，您将": "engagement"
}

ch_df = apply_mapping(ch_df, attr_names_dict, column_pattern='name')
cn_df = apply_mapping(cn_df, attr_names_dict, column_pattern='name')


# %% restructure data

dfs = [ch_df, cn_df]
long_dfs = []

for df in dfs:
    df_long = reshape_conjoint_to_long(df, respondent_id_col="id")
    
    used_cols = [col for col in df.columns if re.match(r"c\d+_atr\d+_(name|p1|p2)", col)]
    conjoint_cols = [col for col in df.columns if "_conjoint_" in col]
    meta_cols = [col for col in df.columns if col not in used_cols + conjoint_cols]
    
    cols_to_keep = meta_cols
    if "id" not in cols_to_keep:
        cols_to_keep.append("id")
    
    df_meta = df[cols_to_keep].drop_duplicates()
    
    df_long = df_long.merge(df_meta, on="id", how="left")
    long_dfs.append(df_long)


# %% replace repeated value before translation

ch_long = long_dfs[0]
cn_long = long_dfs[1]

cn_long['attr_vicinity'] = cn_long['attr_vicinity'].replace({'其他国家': '中国境外'})
ch_long['attr_vicinity'] = ch_long['attr_vicinity'].replace({
    'anderen Ländern': 'einem anderen Land',
    'other countries': 'another country'
})



# %% translate attribute levels

attr_levels_dict = {
    # vicinity
    "einem anderen Land": "abroad",
    "autres pays": "abroad",
    "another country": "abroad",
    "中国境外": "abroad",

    "einem anderen Kanton": "another region",
    "autres cantons": "another region",
    "other cantons": "another region",
    "其他省": "another region",

    "ihrem Kanton": "your region",
    "votre canton": "your region",
    "your canton": "your region",
    "您的省": "your region",

    "ihrer Gemeinde": "your municipality",
    "votre commune": "your municipality",
    "your municipality": "your municipality",
    "您的市": "your municipality",

    # reason
    "in der Nähe der Emissionsquelle liegt": "close to source",
    "est proche de la source d'émission": "close to source",
    "is close to the emission source": "close to source",
    "靠近排放源": "close to source",

    "geringere Kosten als andere Standorte verursacht": "cost-efficient",
    "a un coût inférieur à d'autres lieux": "cost-efficient",
    "has lower cost than other locations": "cost-efficient",
    "比其他地方成本低": "cost-efficient",

    "von dicht besiedelten Gebieten entfernt liegt": "sparsely-populated",
    "est éloigné des zones habitées": "sparsely-populated",
    "is distant from populated areas": "sparsely-populated",
    "远离人口密集区": "sparsely-populated",
    
    # source
    "der Schweiz": "domestic",
    "de Suisse": "domestic",
    "Switzerland": "domestic",
    "中国": "domestic",

    "anderen Ländern": "foreign",
    "d’autres pays": "foreign",
    "other countries": "foreign",
    "其他国家": "foreign",

    # purpose
    "die Reduzierung der Emissionen in der Schweiz": "domestic",
    "de réduire les émissions de dioxyde de carbone en Suisse": "domestic",
    "reducing Switzerland emissions": "domestic",
    "减少中国二氧化碳排放": "domestic",

    "die Speicherung von Emissionen zu Profitzwecken": "foreign",
    "de stocker les émissions pour générer un profit": "foreign",
    "storing emissions for profit": "foreign",
    "通过储存二氧化碳获利": "foreign",
    
    # industry
    "Müllverbrennungsanlagen": "waste incineration",
    "usine d'incinération des déchets": "waste incineration",
    "waste-incineration plant": "waste incineration",
    "垃圾焚烧厂": "waste incineration",

    "Zement-, Stahl- oder Aluminiumwerken": "metal and cement production",
    "cimenterie, aciérie ou aluminerie": "metal and cement production",
    "cement,steel,or aluminum plant": "metal and cement production",
    "水泥厂、钢厂或铝厂": "metal and cement production",

    "Gasbefeuerten Kraftwerken": "gas with CCS",
    "centrale électrique au gaz": "gas with CCS",
    "gas-fired power plant": "gas with CCS",
    "燃气发电厂": "gas with CCS",

    # costs
    "die verschmutzenden Industrie": "polluting industry",
    "les industries polluantes": "polluting industry",
    "polluting industry": "polluting industry",
    "污染企业": "polluting industry",

    "die Allgemeinheit": "taxpayer",
    "tout le monde": "taxpayer",
    "taxpayers": "taxpayer",
    "所有人": "taxpayer",

    # engagement
    "über die Genehmigung oder Ablehnung des Speicherprojekts abstimmen": "vote",
    "voter sur la décision d'approuver ou de rejeter le projet de stockage": "vote",
    "vote on the decision to approve or reject the storage project": "vote",
    "可以对批准或反对存储项目的决定进行表决": "vote",

    "konsultiert, um die Gestaltung des Speicherprojekts mitzubestimmen.": "consult",
    "être consulté pour contribuer à la conception du projet de stockage": "consult",
    "be consulted to help shape the storage project’s design": "consult",
    "接受咨询可以共同制定存储项目的设计": "consult",

    "nur Informationen über die Auswirkungen des Projekts erhalten, aber nicht aktiv an der Entscheidungsfindung teilnehmen können": "inform",
    "recevoir uniquement des informations sur les impacts du projet, mais ne pas participer activement à la prise de décision": "inform",
    "only receive information about the project impacts, but cannot actively participate in decision-making": "inform",
    "仅接收有关项目影响的信息，但不能积极参与决策": "inform"
}

ch_long = apply_mapping(ch_long, attr_levels_dict, column_pattern='attr')
cn_long = apply_mapping(cn_long, attr_levels_dict, column_pattern='attr')

# %% save to file

ch_long.to_csv("data/data_translated_ch.csv", index = False)
cn_long.to_csv("data/data_translated_cn.csv", index = False)

# %% make data file for HCM

values = pd.read_csv("data/data_values_ch_cn.csv")

long_columns = [
    'id', 'task', 'package', 'chosen_plan', 'chosen', 'supported',
    'framing', 'attr_engagement', 'attr_vicinity', 'attr_industry',
    'attr_costs', 'attr_reason', 'attr_source_purpose',
    'age', 'gender', 'ccs_heard', 'ccs_support', 'ccs_important'
]

values_columns = [
    'lreco_1', 'lreco_2', 'lreco_3',
    'galtan_1', 'galtan_2', 'net_zero_question',
    'socio_ecological_1', 'socio_ecological_2',
    'climate_worried', 'id', 'country'
]

ch_filtered = ch_long[long_columns].copy()
cn_filtered = cn_long[long_columns].copy()
values_filtered = values[values_columns].copy()

long_df = pd.concat([ch_filtered, cn_filtered], axis=0)
combined_df = (
    long_df
    .merge(values_filtered, on='id', how='left')
    .rename(columns={
        'net_zero_question': 'galtan_3',
        'climate_worried': 'socio_ecological_3'
    })
)

# %% save data file for HCM

combined_df.to_csv("data/hcm_input.csv", index = False)