# Differentially private training of large language models


## Notes

### Registering new models

To register a new model from the HF hub, use the `/scripts/upload-hf-model.py`.
This script will ensure that additional metadata is saved.

The metadata is needed for some evaluation tasks (mainly MT-Bench).
MT-Bench has a set of standardized model specific prompts and needs to know what base model a fine-tuned model is derived from.
