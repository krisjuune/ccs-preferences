import pandas as pd
import re


# %% import data

pd.set_option('display.max_columns', None)

files = {
    "CH": "raw_data/ccs_conjoint_CH_240225_1004.csv",
    "CN": "raw_data/ccs_conjoint_CN_240225_1752.csv"
}

dataframes = {}

for country, file_name in files.items():
    # read the file, skipping the first two rows
    df = pd.read_csv(file_name, skiprows=[1, 2])
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    
    # store the processed dataframe in the dictionary
    dataframes[country] = df

# %% filter data

# survey launch time
cutoff = pd.Timestamp("2025-02-13 10:00:00")

for country, df in dataframes.items():
    # filter out rows where DistributionChannel is 'preview' or screened_out is True
    df = df[(df['DistributionChannel'] != 'preview') & (df['Finished'] != False)]
    df = df[(df['Q_TerminateFlag'] != "QuotaMet")]
    df = df[(df['Q_TerminateFlag'] != "Screened")]

    # filter out testing IDs using the cutoff time of official launch
    df = df[df["StartDate"] >= cutoff]
    
    # update the dataframe in the dictionary
    dataframes[country] = df

# %% fix typos and delete empty columns

for country, df in dataframes.items():
    cols_to_drop = [
        col for col in df.columns
        if col.startswith("Recipient")
        or col.startswith("Location")
        or col.startswith("Unnamed")
        or col == "IPAddress"
    ]
    dataframes[country] = df.drop(columns=cols_to_drop)

def fix_conjoint_column_names(df):
    new_columns = {}

    for col in df.columns:
        match = re.match(r"^(\d+)(_conjoint_.*)", col)
        if match:
            old_task_num, rest = match.groups()
            new_task_num = int(old_task_num) - 5
            if new_task_num > 0:
                new_col_name = f"{new_task_num}{rest}"
                new_columns[col] = new_col_name

    return df.rename(columns=new_columns)

dataframes["CH"] = fix_conjoint_column_names(dataframes["CH"])

# %% add response IDs and other columns

total_rows = sum(len(df) for df in dataframes.values())
id_counter = 1

for country, df in dataframes.items():
    df['id'] = range(id_counter, id_counter + len(df))
    id_counter += len(df)
    dataframes[country] = df

ch_df = dataframes["CH"]
cn_df = dataframes["CN"]

# %% save clean data

ch_df.to_csv("data/data_clean_untranslated_ch.csv")
cn_df.to_csv("data/data_clean_untranslated_cn.csv")


# %%
