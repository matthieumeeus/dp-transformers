#!/bin/bash

# Define the array of mia_method values
mia_methods=("tfidf_1" "tfidf_5" "tfidf_10" "tfidf_25"
             "jaccard_1" "jaccard_5" "jaccard_10" "jaccard_25"
             "levenshtein_1" "levenshtein_5" "levenshtein_10" "levenshtein_25"
             "embedding_1" "embedding_5" "embedding_10" "embedding_25"
             "ngram_loss_1" "ngram_loss_2" "ngram_loss_3" "ngram_loss_4")

# Loop through each mia_method value and launch the job
for method in "${mia_methods[@]}"; do
    echo "Launching the job with mia_method=${method}"
    python estimate_privacy_synthetic.py --config-name finetune_synthetic_Loraext +submit=True ++shared_inference_parameters.mia_method=${method} ++game_config.seed=126
done

echo "All jobs launched completed."
