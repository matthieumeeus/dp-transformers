#!/bin/bash

python estimate_privacy_synthetic.py --config-name synthetic_agnews_prefix_canary +submit=True \
    +name=synthetic_agnews_prefix_30_canary_1 \
    ++canary_config.prefix_length=30 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_agnews_prefix_canary +submit=True \
    +name=synthetic_agnews_prefix_30_canary_2 \
    ++canary_config.prefix_length=30 \
    ++canary_config.min_ppl=90.0 \
    ++canary_config.max_ppl=110.0 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

echo "All jobs launched completed."