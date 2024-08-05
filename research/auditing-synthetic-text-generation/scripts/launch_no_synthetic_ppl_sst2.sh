#!/bin/bash

# Define the array of parameters
min_ppls=(9.0 28.46 90.0 284.60 900.0 2846.05 9000.0 28460.50 90000.0)
max_ppls=(11.0 34.79 110.0 347.85 1100.0 3478.51 11000.0 34785.05 110000.0)
min_temps=(0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8)
max_temps=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)

# Loop through each config and its corresponding additional parameter
for i in "${!min_ppls[@]}"; do
    echo "Launching the job with min_ppl=${min_ppls[$i]} max_ppl=${max_ppls[$i]} min_temp=${min_temps[$i]} max_temp=${max_temps[$i]}"
    python estimate_privacy_black_box_model_access.py --config-name no_synthetic_sst2_ppl_canary +submit=True ++canary_config.min_ppl=${min_ppls[$i]} ++canary_config.max_ppl=${max_ppls[$i]} ++canary_config.min_temperature=${min_temps[$i]} ++canary_config.max_temperature=${max_temps[$i]}
done

echo "All jobs launched completed."

