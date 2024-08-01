#!/bin/bash

configs=("synthetic_sst2_syntheticcanary_canarylabel" "synthetic_sst2_syntheticcanary_uniformlabel"
         "synthetic_sst2_incanary" "synthetic_agnews_syntheticcanary_canarylabel" 
         "synthetic_agnews_syntheticcanary_uniformlabel" "synthetic_agnews_incanary")

# Define the array of mia_method values
mia_methods=("jaccard_25" "embedding_25" "ngram_2")

# Loop through each mia_method value and launch the job
for config in "${configs[@]}"; do
    for method in "${mia_methods[@]}"; do
        echo "Launching the job with config=${config} and mia_method=${method}"
        python estimate_privacy_synthetic.py --config-name ${config} +submit=True ++shared_inference_parameters.mia_method=${method}
    done
done

echo "All jobs launched completed."
