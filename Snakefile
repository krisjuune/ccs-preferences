configfile: "config.yaml"


rule all:
    input:
        "output/plots/supp_figs/partworths.png",
        "output/plots/country_total.png",
        "output/plots/theta_forest.png",
        "output/plots/supp_figs/theta_heatmap.png",
        "output/plots/value_distributions.png",
        "output/plots/supp_figs/factor_loadings.png",
        "output/plots/predictions_proximity.png",
        "output/plots/predictions_design.png",
        "output/plots/predictions_values.png",
        "output/tables/sample_description.tex",
        "output/tables/model_diagnostics.tex",
        "output/tables/country_gap_decomposition.tex",
        "output/tables/value_correlations_ch.csv",
        "output/tables/value_correlations_cn.csv",
        "output/plots/supp_figs/value_correlations.png",
        (
            "output/data/inference_base_hybrid_choice.nc"
            if config.get("run_base_model", False) else []
        ),
        (
            "output/data/inference_full_interaction_choice.nc"
            if config.get("run_full_interaction_model", False) else []
        ),
        (
            "output/data/inference_main_choice_no_values.nc"
            if config.get("run_no_values_model", False) else []
        ),


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


# Optional, off by default — base HCM without the interaction term, kept for
# comparison against main_hybrid_choice_model below.
if config.get("run_base_model", False):
    rule base_hybrid_choice_model:
        input:
            "data/hcm_input.csv",
        output:
            "output/data/inference_base_hybrid_choice.nc",
        script:
            "scripts/analysis/base_hybrid_choice_model.py"


# Main model: the HCM plus the proximity x Source/Purpose x country
# three-way interaction. Always runs — this is what all the main plots use.
rule main_hybrid_choice_model:
    input:
        "data/hcm_input.csv",
    output:
        "output/data/inference_main_hybrid_choice.nc",
    script:
        "scripts/analysis/main_hybrid_choice_model.py"


# Optional, off by default — H4 comparator: identical to
# main_hybrid_choice_model but without value moderation (no theta terms, no
# SEM measurement model). Used only to compare gamma[country] against the
# main model's gamma, isolating how much adding individual-level values
# changes the estimated country effect.
if config.get("run_no_values_model", False):
    rule main_choice_model_no_values:
        input:
            "data/hcm_input.csv",
        output:
            "output/data/inference_main_choice_no_values.nc",
        script:
            "scripts/analysis/main_choice_model_no_values.py"


rule postprocess_interactions:
    input:
        idata="output/data/inference_main_hybrid_choice.nc",
        hcm_input="data/hcm_input.csv",
    output:
        coefs="output/data/posteriors_interact_coefs.csv",
        conditional="output/data/posteriors_interact_conditional.csv",
    script:
        "scripts/postprocessing/postprocess_interactions.py"


# Scenario-based posterior-predictive probabilities for the paper's four
# hypotheses (H1 inclusive design, H2 proximity x ecol, H3 cost/source x
# values). H4 is addressed separately by rule country_gap_decomposition.
rule predict_scenarios:
    input:
        "output/data/inference_main_hybrid_choice.nc",
    output:
        csv="output/predictions/prediction_scenarios.csv",
        nc="output/predictions/prediction_scenarios.nc",
        txt="output/predictions/prediction_scenarios.txt",
    script:
        "scripts/postprocessing/predict_scenarios.py"


if config.get("run_full_interaction_model", False):
    rule full_interaction_choice_model:
        input:
            "data/hcm_input.csv",
        output:
            "output/data/inference_full_interaction_choice.nc",
        script:
            "scripts/analysis/full_interaction_choice_model.py"


rule postprocess:
    input:
        bcm=(
            "output/data/inference_basic_choice.nc"
            if config.get("run_basic_model", True) else []
        ),
        hcm="output/data/inference_main_hybrid_choice.nc",
        hcm_input="data/hcm_input.csv",
    output:
        beta="output/data/posteriors_beta.csv",
        country="output/data/posteriors_country.csv",
        country_total="output/data/posteriors_country_total.csv",
        theta="output/data/posteriors_theta.csv",
        loadings="output/data/posteriors_loadings.csv",
        alpha="output/data/posteriors_alpha.csv",
        alpha_individual="output/data/posteriors_alpha_individual.csv",
    script:
        "scripts/postprocessing/postprocessing.py"


rule plot_partworths:
    input:
        "output/data/posteriors_beta.csv",
    output:
        "output/plots/supp_figs/partworths.png",
    shell:
        "Rscript scripts/visualisation/plot_partworths.R"


rule plot_country_total:
    input:
        "output/data/posteriors_country_total.csv",
        "output/data/posteriors_interact_conditional.csv",
    output:
        "output/plots/country_total.png",
    shell:
        "Rscript scripts/visualisation/plot_country_total.R"


rule plot_predictions:
    input:
        "output/predictions/prediction_scenarios.csv",
    output:
        "output/plots/predictions_proximity.png",
        "output/plots/predictions_design.png",
        "output/plots/predictions_values.png",
    shell:
        "Rscript scripts/visualisation/plot_predictions.R"


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
        "output/plots/supp_figs/theta_heatmap.png",
    shell:
        "Rscript scripts/visualisation/plot_theta_heatmap.R"


rule plot_value_distributions:
    input:
        "data/data_values_ch_cn.csv",
    output:
        "output/plots/value_distributions.png",
    shell:
        "Rscript scripts/visualisation/plot_value_distributions.R"


rule plot_loadings:
    input:
        "output/data/posteriors_loadings.csv",
    output:
        "output/plots/supp_figs/factor_loadings.png",
    shell:
        "Rscript scripts/visualisation/plot_factor_loadings.R"


rule model_diagnostics:
    input:
        "output/data/inference_main_hybrid_choice.nc",
    output:
        "output/tables/model_diagnostics.tex",
    script:
        "scripts/tables/model_diagnostics.py"


# H4: how much of the Switzerland-China gap in proximity preference is
# compositional (explained by differing value distributions) vs. residual.
rule country_gap_decomposition:
    input:
        country="output/data/posteriors_country.csv",
        country_total="output/data/posteriors_country_total.csv",
    output:
        "output/tables/country_gap_decomposition.tex",
    script:
        "scripts/tables/country_gap_decomposition.py"


rule sample_description:
    input:
        ch="data/data_translated_ch.csv",
        cn="data/data_translated_cn.csv",
    output:
        "output/tables/sample_description.tex",
    shell:
        "python scripts/tables/sample_description.py"


rule value_correlations:
    input:
        ch="data/data_translated_ch.csv",
        cn="data/data_translated_cn.csv",
        values="data/data_values_ch_cn.csv",
    output:
        ch="output/tables/value_correlations_ch.csv",
        cn="output/tables/value_correlations_cn.csv",
    script:
        "scripts/analysis/value_correlations.py"


rule plot_value_correlations:
    input:
        ch="output/tables/value_correlations_ch.csv",
        cn="output/tables/value_correlations_cn.csv",
    output:
        "output/plots/supp_figs/value_correlations.png",
    shell:
        "Rscript scripts/visualisation/plot_value_correlations.R"
