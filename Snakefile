configfile: "config.yaml"


rule all:
    input:
        "output/data/inference_basic_choice.nc",
        "output/data/inference_hybrid_choice.nc",


rule preprocess:
    input:
        ch=config["raw_data"]["ch"],
        cn=config["raw_data"]["cn"],
    output:
        ch="data/data_untranslated_ch.csv",
        cn="data/data_untranslated_cn.csv",
    script:
        "scripts/preprocessing/prepocessing_basics.py"


rule value_indices:
    input:
        ch="data/data_untranslated_ch.csv",
        cn="data/data_untranslated_cn.csv",
    output:
        "data/data_values_ch_cn.csv",
    shell:
        "python scripts/preprocessing/value_indices.py"


rule translate_conjoints:
    input:
        ch="data/data_untranslated_ch.csv",
        cn="data/data_untranslated_cn.csv",
        values="data/data_values_ch_cn.csv",
    output:
        ch="data/data_translated_ch.csv",
        cn="data/data_translated_cn.csv",
        hcm="data/hcm_input.csv",
    shell:
        "python scripts/preprocessing/translate_conjoints.py"


rule basic_choice_model:
    input:
        "data/hcm_input.csv",
    output:
        "output/data/inference_basic_choice.nc",
    script:
        "scripts/analysis/basic_choice_model.py"


rule hybrid_choice_model:
    input:
        "data/hcm_input.csv",
    output:
        "output/data/inference_hybrid_choice.nc",
    script:
        "scripts/analysis/hybrid_choice_model.py"
