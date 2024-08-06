#!/bin/bash

# Define the array of mia_method values
multiples=(1 2 4 8)

# Define the array of mia_method values
# Note: probably want to do this first for one end-to-end and then for all to recycle the main components
mia_methods=("jaccard_25" "embedding_25" "ngram_2")

# Loop through each mia_method value and launch the job
for multiple in "${multiples[@]}"; do
    for method in "${mia_methods[@]}"; do
        echo "Launching the job with multiple=${multiple} and mia_method=${method}"
        python estimate_privacy_synthetic.py --config-name synthetic_sst2_syntheticcanary_uniformlabel +submit=True ++shared_training_parameters.synthetic_multiple=${multiple} ++shared_inference_parameters.mia_method=${method}
    done
done

echo "All jobs launched completed."
