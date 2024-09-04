#!/bin/bash

# Define the array of parameters
min_ppls=(9.0 28.46 90.0 284.60 900.0)
max_ppls=(11.0 34.79 110.0 347.85 1100.0)
min_temps=(0.8 0.8 0.8 0.8 0.8)
max_temps=(1.0 1.0 1.0 1.0 1.0)

# Loop through each config and its corresponding additional parameter
for i in "${!min_ppls[@]}"; do
    echo "Launching the job with min_ppl=${min_ppls[$i]} max_ppl=${max_ppls[$i]} min_temp=${min_temps[$i]} max_temp=${max_temps[$i]}"
    python estimate_privacy_synthetic.py --config-name synthetic_sst2_prefix_canary +submit=True +name=synthetic_sst2_prefix_20_canary_$i \
        ++canary_config.prefix_length=20 \
        ++canary_config.min_ppl=${min_ppls[$i]} \
        ++canary_config.max_ppl=${max_ppls[$i]} \
        ++canary_config.min_temperature=${min_temps[$i]} \
        ++canary_config.max_temperature=${max_temps[$i]}
done

echo "All jobs launched completed."
