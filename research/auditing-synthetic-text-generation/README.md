# Auditing Synthetic Text Generation

```
az ml job create -f ./sst2.yml --web
```

If you get a not found error or a not authorized error see https://dev.azure.com/ii-m365/M365Research/_wiki/wikis/wiki/6/azure_machine_learning?anchor=%60az-ml%60-issue-when-submitting-an-experiment


## Full auditing pipeline

### Environment

``` bash
pip install git+https://github.com/microsoft/responsible-ai-toolbox-privacy.git@717badca929f9c1e774660d9be3e66e9434d34ae#egg=privacy_estimates[pipelines]
```

**Note:** Please upgrade the package regularly as it is under active development

To update to the latest version, run the following command:

``` bash
pip uninstall privacy_estimates; pip install git+https://github.com/microsoft/responsible-ai-toolbox-privacy.git@717badca929f9c1e774660d9be3e66e9434d34ae#egg=privacy_estimates[pipelines]
```

### Run the auditing pipeline

#### Threat model: Black box access

This threat model assumes direct access to the model's predictions.
The model was trained on the sensitive data without a synthetic data generation step.
The pipeline uses RMIA scores (computed using the likelihood predicted by the model for the target canary in high precision). 

``` bash
python estimate_privacy_black_box_model_access.py --config-name no_synthetic_sst2_externalcanary_canarylabel +submit=True
```

#### Threat model: Synthetic data only

This threat model assumes solely access to the generated synthetic data from the target model. 
We allow for multiple membership signals to be used in the RMIA setup, to be specified by `shared_inference_parameters.mia_method` (by default the best attack using 2-gram likelihood). 

``` bash
python estimate_privacy_synthetic.py --config-name synthetic_sst2_externalcanary_canarylabel +submit=True
```

#### Overview of the canary creation options

We allow for multiple canary generation and injection mechansisms. 

- `canary_method` describes the overall **canary text** generation. Different options are:
    - `hold_out_original_data`: canary text is randomly selected from the training data (in-distribution) of exactly `canary_length` words. We ensure there is then no overlap with the other training or validation data. Note that the number of words that is feasible depends on the dataset used (e.g. for AgNews 50 words is feasible, but 100 is not for 1000 canaries). 
    - `sample_real`: canary text is sampled from an external, out-of-distribution dataset to be provided with `external_artifact` and `external_artifact_version`. Note that we need to specifiy the `canary_text_column`. 
    - `sample_synthetic`: canary text is synthetically generated using the model specified by  `external_artifact` and `external_artifact_version`. By default we apply rejection sampling until we have a sufficient amount of canaries of the required length and with a perplexity between `min_ppl` and `max_ppl`. Perplexity is computed using the prompt with the right label (see label compatibility below). The temperature is automatically adapted to converge to the target perplexity range, initialized with `min_temperature` and `max_temperature`. Some edge cases:
        - When `min_ppl`==`max_ppl`, we do not control for perplexity and just sample from the model using the temperature (`min_temperature` / `max_temperature` ) / 2. 
        - When `max_ppl` == -1, we sample random tokens from the vocabulary. 
- `label_comptability_method` describes how the cvanary text should be made compatible with the labels of the training dataset. We have two options:
    - 'uniform': sample random labels from the training dataset, ensuring the label distribution matches. 
    - 'extend': extend the label distribution with a canary-specific label, by default 'canary'. 
- We further provide a way to replace tokens from the canary text by either using a masked language model or random replacement. When `num_tokens_to_replace`==0, nothing happens. 
