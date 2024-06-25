# Auditing Synthetic Text Generation

```
az ml job create -f ./sst2.yml --web
```

If you get a not found error or a not authorized error see https://dev.azure.com/ii-m365/M365Research/_wiki/wikis/wiki/6/azure_machine_learning?anchor=%60az-ml%60-issue-when-submitting-an-experiment


## Full auditing pipeline

### Environment

``` bash
pip install git+https://github.com/microsoft/responsible-ai-toolbox-privacy.git@1669e787806f0e43edff3c9c336203646ad49d00#egg=privacy_estimates[pipelines]
```


**Note:** Please upgrade the package regularly as it is under active development

To update to the latest version, run the following command:

``` bash
pip uninstall privacy_estimates; pip install git+https://github.com/microsoft/responsible-ai-toolbox-privacy.git@1669e787806f0e43edff3c9c336203646ad49d00#egg=privacy_estimates[pipelines]
```

### Run the auditing pipeline

#### Threat model: Black box access

This threat model assumes direct access to the model's predictions.
The model was trained on the sensitive data without a synthetic data generation step.

``` bash
python estimate_privacy_black_box_model_access.py --config-name peft_sft +submit=True
```
