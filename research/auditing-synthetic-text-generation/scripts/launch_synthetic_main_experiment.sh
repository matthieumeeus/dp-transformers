#!/bin/bash

# Define the array of mia_method values
configs=("synthetic_sst2_syntheticcanary_canarylabel" "synthetic_sst2_syntheticcanary_uniformlabel"
         "synthetic_sst2_incanary" "synthetic_agnews_syntheticcanary_canarylabel" 
         "synthetic_agnews_syntheticcanary_uniformlabel" "synthetic_agnews_incanary")

# Loop through each mia_method value and launch the job
for config in "${configs[@]}"; do
    echo "Launching the job with config=${config}"
    python estimate_privacy_synthetic.py --config-name ${config} +submit=True 
done

echo "All jobs launched completed."
