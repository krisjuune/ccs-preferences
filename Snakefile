configfile: "config.yaml"


rule all:
    input:
        "output/plots/partworths.png",
        "output/plots/country_utilities.png",
        "output/plots/theta_forest.png",
        "output/plots/theta_heatmap.png",


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


rule postprocess:
    input:
        bcm="output/data/inference_basic_choice.nc",
        hcm="output/data/inference_hybrid_choice.nc",
    output:
        beta="output/data/posteriors_beta.csv",
        country="output/data/posteriors_country.csv",
        theta="output/data/posteriors_theta.csv",
    script:
        "scripts/postprocessing/postprocessing.py"


rule plot_partworths:
    input:
        "output/data/posteriors_beta.csv",
    output:
        "output/plots/partworths.png",
    shell:
        "Rscript scripts/visualisation/plot_partworths.R"


rule plot_country:
    input:
        "output/data/posteriors_country.csv",
    output:
        "output/plots/country_utilities.png",
    shell:
        "Rscript scripts/visualisation/plot_country.R"


rule plot_theta_forest:
    input:
        "output/data/posteriors_theta.csv",
    output:
        "output/plots/theta_forest.png",
    shell:
        "Rscript scripts/visualisation/plot_theta_forest.R"


rule plot_theta_heatmap:
    input:
        "output/data/posteriors_theta.csv",
    output:
        "output/plots/theta_heatmap.png",
    shell:
        "Rscript scripts/visualisation/plot_theta_heatmap.R"
