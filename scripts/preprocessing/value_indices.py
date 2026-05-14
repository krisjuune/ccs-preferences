import pandas as pd
import numpy as np


def apply_mapping(df, mapping_dict, column_pattern=None):
    if isinstance(column_pattern, str):
        column_patterns = [column_pattern]
    elif isinstance(column_pattern, list) and all(isinstance(pat, str) for pat in column_pattern):
        column_patterns = column_pattern
    elif column_pattern is None:
        column_patterns = []
    else:
        raise ValueError("column_pattern should be a string, list of strings, or None.")

    if column_patterns:
        columns_to_map = [col for col in df.columns if any(pat in col for pat in column_patterns)]
    else:
        columns_to_map = df.columns

    for column in columns_to_map:
        df[column] = df[column].replace(mapping_dict)

    return df

# %%

files = {
    "switzerland": "data/data_untranslated_ch.csv",
    "china": "data/data_untranslated_cn.csv"
}


dataframes = {}

for country, file_name in files.items():
    df = pd.read_csv(file_name)
    dataframes[country] = df


# %% values

socio_econ_list = [
    "lreco_1",
    "lreco_2",
    "lreco_3",
]

socio_cult_list = [
    "galtan_1",
    "galtan_2",
    "net_zero_question"
]

socio_ecol_list = [
    "socio_ecological_1",
    "socio_ecological_2",
    "climate_worried"
]

value_columns = socio_econ_list + socio_cult_list + socio_ecol_list

reversed_scale_list = [
    "lreco_1", 
    "galtan_1",
    "galtan_2",
    "net_zero_question",
    "socio_ecological_1"
]

five_point_reversed_list = [
    "climate_worried"
]

likert_values_value_ques = [
    'Completely disagree',
    'Somewhat disagree',
    'Disagree',
    'Somewhat agree',
    'Agree',
    'Completely agree'
]

net_zero_translation = {
    "Absolutely sufficient":"Completely agree",
    "Sufficient":"Agree",
    "Slightly sufficient":"Somewhat agree",
    "Slightly insufficient":"Somewhat disagree",
    "Insufficient":"Disagree",
    "Absolutely insufficient":"Completely disagree"
}

for country, df in dataframes.items():
    df = apply_mapping(df, net_zero_translation, column_pattern='net_zero')
    dataframes[country] = df

likert_values_five = [
    "Not at all worried", 
    "Not very worried",
    "Somewhat worried",
    "Very worried",
    "Extremely worried"
]

numerical_values_normalised = [0, 0.2, 0.4, 0.6, 0.8, 1]
numerical_values_reversed =  [1, 0.8, 0.6, 0.4, 0.2, 0]
numerical_values_five = [1, 0.75, 0.5, 0.25, 0]

values_dict = {**dict(np.array(list(zip(likert_values_value_ques, numerical_values_normalised))))}
values_dict_reversed = {**dict(np.array(list(zip(likert_values_value_ques, numerical_values_reversed))))}
values_dict_five = {**dict(np.array(list(zip(likert_values_five, numerical_values_five))))}

def get_values_dict(column_name):
    if column_name in reversed_scale_list:
        return values_dict_reversed
    elif column_name in five_point_reversed_list:
        return values_dict_five
    else:
        return values_dict

for country, df in dataframes.items():
    # translate net zero question
    df = apply_mapping(df, net_zero_translation, column_pattern='net_zero')
    
    for col in value_columns:
        if col in df.columns:
            df = apply_mapping(df, get_values_dict(col), column_pattern=col)
            df[col] = df[col].replace(["Not sure", "Prefer not to say"], np.nan)

    if set(socio_econ_list).issubset(df.columns):
        df['lreco'] = df[socio_econ_list].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        df.loc[df[socio_econ_list].isnull().any(axis=1), 'lreco'] = np.nan
    if set(socio_cult_list).issubset(df.columns):
        df['galtan'] = df[socio_cult_list].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        df.loc[df[socio_cult_list].isnull().any(axis=1), 'galtan'] = np.nan
    if set(socio_ecol_list).issubset(df.columns):
        df['socio_ecol'] = df[socio_ecol_list].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        df.loc[df[socio_ecol_list].isnull().any(axis=1), 'socio_ecol'] = np.nan

    dataframes[country] = df

value_columns = value_columns + ["lreco", "galtan", "socio_ecol"]

value_data = []

for country, df in dataframes.items():
    cols = value_columns + ["id"]
    df_selected = df[cols].copy()
    df_selected['country'] = country
    value_data.append(df_selected)

value_data = pd.concat(value_data, ignore_index=True)

# %% save value data 

value_data.to_csv("data/data_values_ch_cn.csv", index = False)

# %%
