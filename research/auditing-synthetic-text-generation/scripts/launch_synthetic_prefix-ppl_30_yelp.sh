#!/bin/bash

python estimate_privacy_synthetic.py --config-name synthetic_yelp_prefix_canary +submit=True \
    +name=synthetic_yelp_prefix_0_canary_ppl_31 \
    ++canary_config.prefix_length=0 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_yelp_prefix_canary +submit=True \
    +name=synthetic_yelp_prefix_10_canary_ppl_31 \
    ++canary_config.prefix_length=10 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_yelp_prefix_canary +submit=True \
    +name=synthetic_yelp_prefix_20_canary_ppl_31 \
    ++canary_config.prefix_length=20 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_yelp_prefix_canary +submit=True \
    +name=synthetic_yelp_prefix_30_canary_ppl_31 \
    ++canary_config.prefix_length=30 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

echo "All jobs launched completed."