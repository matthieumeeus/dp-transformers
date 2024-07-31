# Auditing Synthetic Text Generation

## (1) Environment

``` bash
pip install git+https://github.com/microsoft/responsible-ai-toolbox-privacy.git@717badca929f9c1e774660d9be3e66e9434d34ae#egg=privacy_estimates[pipelines]
```

**Note:** Please upgrade the package regularly as it is under active development

To update to the latest version, run the following command:

``` bash
pip uninstall privacy_estimates; pip install git+https://github.com/microsoft/responsible-ai-toolbox-privacy.git@717badca929f9c1e774660d9be3e66e9434d34ae#egg=privacy_estimates[pipelines]
```

## (2) Understanding the config

#### 2.1 Model training

Across all experiments, we consider the exact same regime to train the target (and thus also reference) models. Some things to keep in mind:
- We use the templated prompt for training, with the corresponding label filled out. This means that the attention mask is set to 1 on the prompt tokens, but that their labels are set to -100 and that there is thus no backpropagation for the prompt - only for the completion. 
- We train the model for 1 epoch, with overall batch size of 16, sequence length 256 (mostly for canaries in the case of sst-2), learning rate of 2e-5. We use LoRA with all target modules, dimension of r=4, bf16 as dtype and no quantization. By default we also use gradient checkpointing, and we set both bf16 and fp16 to False. 

#### 2.2 Inference

As part of the inference component, we compute a membership signal for each target sequence - which is then further used to compute an RMIA score (combining the signal from the target and reference models). Computing the membership signal differs for each threat model:
- Black-box model access. With the option `expsum` we compute the likelihood of the target sequence predicted by the model. This comes down to a product of conditional probabilities - which becomes extremely small very quickly. Therefore we compute the log of the membership signal first, which is then transformed to the probability again at the level of the RMIA attck (in privacy-estimates). Importantly, we compute the sequence level likelhood in the same way as it is used during training, i.e. with the prompt attention mask equal to 1 and its labels to be ignored.
- Synthetic data attack. We here consider a variety of MIA methods, ranging from training an n-gram model to the mean similarity to the k closest records. Some things to keep in mind:
    - In this case, the training component actually returns synthetic data generated from the finetuned model (while above it return the finetuned model). 
    - The n-gram signal is computed just as above, with a sequence-level likelihood that becomes extremely small very quickly and is propagated to the RMIA level through its log. 
    - The distance based signals also need to be bounded by [0,1], where closer to 1 should correspond to more likely to be a member. Hence, we compute the *mean normalized similarity* to the k closest synthetic sequences. This is 1 when all k closest sequences are the same. By default, we consider jaccard, levenshtein (string space) and cosine similarity (embedding space) as similarity metrics and `k=1,5,10,25`. 
    - By default, we compute all synthetic data membership inference signals - and just select the one specified in `shared_inference_parameters.mia_method`. See 3.2 to easily compute the MIA performance across all synthetic MIA methods. 


#### 2.2 Overview of the canary creation options

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

## (3) Run the auditing pipeline

#### 3.1 Threat model: Black box access

This threat model assumes direct access to the model's predictions.
The model was trained on the sensitive data without a synthetic data generation step.
The pipeline uses RMIA scores (computed using the likelihood predicted by the model for the target canary in high precision). 

``` bash
python estimate_privacy_black_box_model_access.py --config-name no_synthetic_sst2_externalcanary_canarylabel +submit=True
```

For the main experiment (table with MIA performance across attacks and setups), we launched:

``` bash
./scripts/launch_no_synthetic_main_experiment.sh > no_synthetic_main_exp.txt
```

#### 3.2 Threat model: Synthetic data only

This threat model assumes solely access to the generated synthetic data from the target model. 
We allow for multiple membership signals to be used in the RMIA setup, to be specified by `shared_inference_parameters.mia_method` (by default the best attack using 2-gram likelihood). 

``` bash
python estimate_privacy_synthetic.py --config-name synthetic_sst2_externalcanary_canarylabel +submit=True
```

For the main experiment we launched: 

``` bash
./scripts/launch_synthetic_main_experiment.sh > synthetic_main_exp.txt
```

**Other MI signals.** By default, all synthetic membership signals are computed and only one signal is selected to run the attack. However, when the entire pipeline has been run once, we can re-use all computation-heavy components to compute the MIA performance for all other membership signals too. This can be run with a simple bash script:

``` bash
./scripts/launch_synthetic_mias.sh > all_synthetic_jobs.txt
```

Note that we save the output in a txt file, as we will easily extract all job urls from the txt output for further analysis. 

**Vary synthetic multiple.** By default, the target model generates as many synthetic data records as provided in the training dataset. To increase this, we consider the variable `shared_training_parameters.synthetic_multiple`. To run through various variable, we consider the following bash script:

``` bash
./scripts/launch_synthetic_multiples.sh > all_synthetic_multiples.txt
```

When we also want to compute all MIA methods across synthetic multiples, we need to combine both bash scripts above with an nested for loop. 

**Vary perplexity of synthetic canary.** To understand canary vulnerability versus canary perplexity, we need to launch the attack pipeline end-to-end for both the non-synthetic and synthetic attack for different ranges of perplexity. To run through this, we also design a bash script for both:

``` bash
./scripts/launch_no_synthetic_ppl.sh > no_synthetic_ppl_exp.txt
```

``` bash
./scripts/launch_synthetic_ppl.sh > synthetic_ppl_exp.txt
```

Note that we here need to specify the min and max perplexity of the range to be considered, and also need to give to provide an inital min and max temperature to be used in the temperature optimization. 

Importantly, we cannot recycle the trained target/reference models across no-synthetic/synthetic as we use different number of repetitions. 

For comparison, we also run the same attacks (with the same repetitions) for in-distribution canaries using the default commands (3.1 and 3.2) with as config files `no_synthetic_agnews_incanary.yaml` and `synthetic_agnews_incanary.yaml`. 

## (4) Analyze the results

**Get MIA performance.** 

The notebook `notebooks/get_mia_results.ipynb` contains the code to compute the MIA performance (AUC, tpr at low fpr) from a certain executed job and its url. 
It contains (1) just the functionality to get the MIA performance for a given url, (2) computing the MIA performance across a series of jobs launched using a bash script (parsing the urls from a txt file as above) and (3) how to plot the main curve for AgNews. 

**Scatter plots.**

To generate the scatter plots for the disparate vulnerability of canaries, we use `notebooks/scatter.ipynb`. You just need two jobs run for the exact same canaries for two attacks to run this. 

**Perplexity results.** 

First, we save all urls corresponding to the experiments run with varying canary perplexity here in `notebooks/ppl_experiment_urls.json`. This json can be updated when the perplexity ranges we want to consider and the corresponding urls are updated. Next, the code to generate the figure from the results is in `notebooks/ppl_exp_results_from_json.ipynb`. 

## (5) Compute the synthetic data utility

We also need to compute the utility of the synthetic data that is being generated. For this we first compute the utility of the 'real' data, by training a roberta model for classification on the real training data and evaluate on a held-out test set. 

To compute this, launch with DATASET={sst2, agnews}:

``` bash
az ml job create -f ./configs/compute_utility_real_{DATASET}.yml --web
```

Then we also need to run this for a roberta model trained on synthetic data (and still evaluated on the real test data). The drop in performance compared to the real data then indicates the utility of the synthetic data. So launch the following to first generate synthetic data and then evaluate downstream performance:

``` bash
az ml job create -f ./configs/compute_utility_synthetic_{DATASET}.yml --web
```

To analyze the results (get the downstream performance and get plots for the appendix, see `notebooks/viz_utility.ipynb`).