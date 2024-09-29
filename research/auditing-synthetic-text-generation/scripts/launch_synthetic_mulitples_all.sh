#!/bin/bash

python estimate_privacy_synthetic.py --config-name synthetic_sst2_prefix_canary +submit=True \
    +name=synthetic_sst2_multiple_2 \
    ++game_config.num_repetitions=2 \
    ++canary_config.prefix_length=0 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_sst2_prefix_canary +submit=True \
    +name=synthetic_sst2_multiple_4 \
    ++game_config.num_repetitions=4 \
    ++canary_config.prefix_length=0 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_sst2_prefix_canary +submit=True \
    +name=synthetic_sst2_multiple_8 \
    ++game_config.num_repetitions=8 \
    ++canary_config.prefix_length=0 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_sst2_prefix_canary +submit=True \
    +name=synthetic_sst2_multiple_16 \
    ++game_config.num_repetitions=16 \
    ++canary_config.prefix_length=0 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021


echo "All jobs launched completed."