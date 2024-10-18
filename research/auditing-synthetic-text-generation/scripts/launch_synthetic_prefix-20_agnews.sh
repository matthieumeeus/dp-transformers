#!/bin/bash

python estimate_privacy_synthetic.py --config-name synthetic_agnews_prefix_canary +submit=True \
    +name=synthetic_agnews_prefix_20_canary_0 \
    ++canary_config.prefix_length=20 \
    ++canary_config.min_ppl=9.0 \
    ++canary_config.max_ppl=11.0 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_agnews_prefix_canary +submit=True \
    +name=synthetic_agnews_prefix_20_canary_1 \
    ++canary_config.prefix_length=20 \
    ++canary_config.min_ppl=28.46 \
    ++canary_config.max_ppl=34.79 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_agnews_prefix_canary +submit=True \
    +name=synthetic_agnews_prefix_20_canary_2 \
    ++canary_config.prefix_length=20 \
    ++canary_config.min_ppl=90.0 \
    ++canary_config.max_ppl=110.0 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_agnews_prefix_canary +submit=True \
    +name=synthetic_agnews_prefix_20_canary_3 \
    ++canary_config.prefix_length=20 \
    ++canary_config.min_ppl=284.60 \
    ++canary_config.max_ppl=347.85 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.0 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_agnews_prefix_canary +submit=True \
    +name=synthetic_agnews_prefix_20_canary_7 \
    ++canary_config.prefix_length=20 \
    ++canary_config.min_ppl=28460.50 \
    ++canary_config.max_ppl=34785.05 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.4 \
    ++canary_config.seed=19021

python estimate_privacy_synthetic.py --config-name synthetic_agnews_prefix_canary +submit=True \
    +name=synthetic_agnews_prefix_20_canary_8 \
    ++canary_config.prefix_length=20 \
    ++canary_config.min_ppl=90000.0 \
    ++canary_config.max_ppl=110000.0 \
    ++canary_config.min_temperature=0.8 \
    ++canary_config.max_temperature=1.4 \
    ++canary_config.seed=19021

echo "All jobs launched completed."