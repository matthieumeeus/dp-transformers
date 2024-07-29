#!/bin/bash

# Define the array of parameters
min_ppls=(900 4500 9000)
max_ppls=(1100 5500 11000)
min_temps=(0.8 0.8 1.0)
max_temps=(1.1 1.2 1.5)

# Loop through each config and its corresponding additional parameter
for i in "${!min_ppls[@]}"; do
    echo "Launching the job with min_ppl=${min_ppls[$i]} max_ppl=${max_ppls[$i]} min_temp=${min_temps[$i]} max_temp=${max_temps[$i]}"
    python estimate_privacy_black_box_model_access.py --config-name no_synthetic_agnews_pplcanary +submit=True ++canary_config.min_ppl=${min_ppls[$i]} ++canary_config.max_ppl=${max_ppls[$i]} ++canary_config.min_temperature=${min_temps[$i]} ++canary_config.max_temperature=${max_temps[$i]}
done

echo "All jobs launched completed."

