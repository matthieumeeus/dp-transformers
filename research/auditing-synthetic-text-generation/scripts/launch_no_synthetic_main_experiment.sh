#!/bin/bash

# Define the array of mia_method values
configs=("no_synthetic_sst2_syntheticcanary_canarylabel" "no_synthetic_sst2_syntheticcanary_uniformlabel"
         "no_synthetic_sst2_incanary" "no_synthetic_agnews_syntheticcanary_canarylabel" 
         "no_synthetic_agnews_syntheticcanary_uniformlabel" "no_synthetic_agnews_incanary")

# Loop through each mia_method value and launch the job
for config in "${configs[@]}"; do
    echo "Launching the job with config=${config}"
    python estimate_privacy_black_box_model_access.py --config-name ${config} +submit=True 
done

echo "All jobs launched completed."
