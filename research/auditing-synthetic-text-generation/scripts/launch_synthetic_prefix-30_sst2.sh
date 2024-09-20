#!/bin/bash

# Define the array of parameters
min_ppls=(9.0 28.46 90.0 284.60)
max_ppls=(11.0 34.79 110.0 347.85)
min_temps=(0.8 0.8 0.8 0.8 0.8)
max_temps=(1.0 1.0 1.0 1.0 1.0)

python estimate_privacy_synthetic.py --config-name synthetic_sst2_prefix_canary +submit=True \
    +name=synthetic_sst2_prefix_30_canary_0 \
    ++canary_config.prefix_length=30 \
    ++canary_config.min_ppl=9.0 \
    ++canary_config.max_ppl=11.0 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_sst2_prefix_canary +submit=True \
    +name=synthetic_sst2_prefix_30_canary_1 \
    ++canary_config.prefix_length=30 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_sst2_prefix_canary +submit=True \
    +name=synthetic_sst2_prefix_30_canary_2 \
    ++canary_config.prefix_length=30 \
    ++canary_config.min_ppl=90.0 \
    ++canary_config.max_ppl=110.0 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_sst2_prefix_canary +submit=True \
    +name=synthetic_sst2_prefix_30_canary_3 \
    ++canary_config.prefix_length=30 \
    ++canary_config.min_ppl=284.60 \
    ++canary_config.max_ppl=347.85 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021
