#!/bin/bash

# Define the array of mia_method values
multiples=(2 4 8)

# Loop through each mia_method value and launch the job
for multiple in "${multiples[@]}"; do
    echo "Launching the job with multiple=${multiple}"
    python estimate_privacy_synthetic.py --config-name synthetic_agnews_externalcanary_canarylabel +submit=True ++shared_training_parameters.synthetic_multiple=${multiple}
done

echo "All jobs launched completed."
