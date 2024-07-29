#!/bin/bash

# Define the array of parameters
min_ppls=(675 2250)
max_ppls=(825 2750)
min_temps=(0.9 1.0)
max_temps=(1.1 1.2)

# Loop through each config and its corresponding additional parameter
for i in "${!min_ppls[@]}"; do
    echo "Launching the job with min_ppl=${min_ppls[$i]} max_ppl=${max_ppls[$i]} min_temp=${min_temps[$i]} max_temp=${max_temps[$i]}"
    python estimate_privacy_synthetic.py --config-name synthetic_agnews_ppl_canary +submit=True ++canary_config.min_ppl=${min_ppls[$i]} ++canary_config.max_ppl=${max_ppls[$i]} ++canary_config.min_temperature=${min_temps[$i]} ++canary_config.max_temperature=${max_temps[$i]}
done

echo "All jobs launched completed."
