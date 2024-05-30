
import nltk
from nltk.data import find
import Levenshtein
import numpy as np
from tqdm import tqdm
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def download_punkt_if_not_exists():
    try:
        # Check if 'punkt' tokenizer models are already available
        find('tokenizers/punkt')
        print("Punkt tokenizer is already downloaded.")
    except LookupError:
        # If not found, download the 'punkt' tokenizer models
        print("Punkt tokenizer not found. Downloading now...")
        nltk.download('punkt')
        print("Punkt tokenizer downloaded.")

VALID_METHODS = ['n-gram', 'distance_str', 'distance_emb']

### FOR DISTANCE BASED METHODS
def jaccard_distances(sample, synthetic_tokenized):
    set1 = set(nltk.word_tokenize(sample.lower()))
    return [nltk.jaccard_distance(set1, set2) for set2 in synthetic_tokenized]

def levenshtein_distances(sample, synthetic_lower):
    return [Levenshtein.distance(sample.lower(), sentence2) for sentence2 in synthetic_lower]

def tfidf_distances(sample, tfidf_vectorizer, synthetic_tfidf):
    
    # transform the sample too
    tfidf_original = tfidf_vectorizer.transform([sample]).toarray()

    # also get the binary version of the original tfidf
    tfidf_original_binary = tfidf_original.copy()
    tfidf_original_binary[tfidf_original_binary > 0] = 1
    
    # we first want to compute the sum of the importance in the original sentence that is present in each synthetic sequence
    similarities = tfidf_original @ synthetic_tfidf.T

    # but we need to normalize this by how many words contribute to that, ie the union of the words in the original and synthetic
    norm_factors = tfidf_original_binary @ synthetic_tfidf.T

    return [1 - sim / norm_factors[0][j]  if norm_factors[0][j] > 0 else 1 for j, sim in enumerate(similarities[0])]

def compute_str_distance_mia_score(samples, synthetic, 
                     distance_metrics=['jaccard', 'levenshtein', 'tfidf'], 
                     ks = [1, 5, 10, 25]):
    sample_scores = {}

    # do some preprocessing on synthetic
    synthetic_lower = [text.lower() for text in synthetic]
    synthetic_tokenized = [set(nltk.word_tokenize(synthetic_sample)) for synthetic_sample in synthetic_lower]

    # fit and transform tfidf on synthetic data - only keeping (0,1) values
    tfidf_vectorizer = TfidfVectorizer()
    synthetic_tfidf = tfidf_vectorizer.fit_transform(synthetic).toarray()
    synthetic_tfidf[synthetic_tfidf > 0] = 1

    for distance_metric in distance_metrics:
        for i, sample in tqdm(enumerate(samples), desc=f'Computing distance MIA scores for {distance_metric}'):
            if distance_metric == 'tfidf':
                distances = tfidf_distances(sample, tfidf_vectorizer, synthetic_tfidf)
            elif distance_metric == 'jaccard':
                distances = jaccard_distances(sample, synthetic_tokenized)
            elif distance_metric == 'levenshtein':
                distances = levenshtein_distances(sample, synthetic_lower)
            else:
                raise ValueError(f"Invalid distance metric: {distance_metric}")
            distances_sorted = np.sort(distances)
            for k in ks:
                mean_closest_k = np.mean(distances_sorted[:k])
                if i == 0:
                    sample_scores[f'{distance_metric}_{k}'] = [mean_closest_k]
                else:
                    sample_scores[f'{distance_metric}_{k}'].append(mean_closest_k)

    return sample_scores

def compute_emb_distance_mia_score(samples, synthetic, synthetic_embeddings, ks = [1, 5, 10, 25], 
                                   emb_model_name = 'all-mpnet-base-v2'): #'paraphrase-MiniLM-L6-v2'):
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer(emb_model_name)

    if synthetic_embeddings is None:
        print("Computing the embeddings for the synthetic data")
        synthetic_embeddings = model.encode(synthetic)
    
    sample_scores = {}
    sample_embeddings = model.encode(samples)
    embedding_similarities = model.similarity(sample_embeddings, synthetic_embeddings).numpy()

    for i, sample in tqdm(enumerate(samples), desc=f'Computing distance MIA scores for embedding distance'):
        distances = 1 - embedding_similarities[i]
        distances_sorted = np.sort(distances)
        for k in ks:
            mean_closest_k = np.mean(distances_sorted[:k])
            if i == 0:
                sample_scores[f'embedding_{k}'] = [mean_closest_k]
            else:
                sample_scores[f'embedding_{k}'].append(mean_closest_k)

    return sample_scores, synthetic_embeddings

### FOR N-GRAM BASED METHODS

def generate_ngrams(text, n):
    """
    Generate n-grams from the input text.
    """
    tokens = text.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

def train_ngram_model(all_text, n, smoothing=1):
    """
    Train an n-gram model from the given text using Laplace smoothing.
    """
    all_ngrams = []
    vocabulary = set()

    for text in all_text:
        words = text.split()
        vocabulary.update(words)
        ngrams = generate_ngrams(text, n)
        all_ngrams.extend(ngrams)

    ngram_counts = collections.Counter(all_ngrams)
    total_ngrams = sum(ngram_counts.values()) + smoothing * len(vocabulary) ** n

    # Convert counts to probabilities with smoothing
    ngram_probabilities = {
        ngram: (count + smoothing) / total_ngrams
        for ngram, count in ngram_counts.items()
    }

    return ngram_probabilities, len(vocabulary)

def compute_loss(ngram_model, text, n, vocabulary_size, smoothing=1):
    """
    Compute the loss of the n-gram model on a given piece of text.
    The loss is the average negative log likelihood of the n-grams in the text.
    """
    ngrams = generate_ngrams(text, n)
    log_likelihood = 0
    count = 0

    for ngram in ngrams:
        if ngram in ngram_model:
            prob = ngram_model[ngram]
        else:
            # Apply smoothing for unseen n-grams
            prob = smoothing / (sum(ngram_model.values()) + smoothing * vocabulary_size ** n)
        log_likelihood += np.log(prob)
        count += 1

    loss = -log_likelihood / count if count > 0 else float('inf')

    return loss

def compute_ngram_mia_score(samples, synthetic, ns = [1, 2, 3, 4]):
    sample_scores = {}
    for n in ns:
        print(f"Training the {n}-gram model on and computing its losses.")
        all_text = synthetic
        ngram_model, vocab_size = train_ngram_model(all_text, n)
        sample_scores[f'ngram_loss_{n}'] = [compute_loss(ngram_model, sample, n, vocab_size) for sample in samples]

    return sample_scores

def compute_mia_score(samples, synthetic, method, synthetic_embeddings = None):
    all_mia_scores = {}
    if method == 'all':
        methods = VALID_METHODS
    for method in methods:
        if method == 'n-gram':
            scores = compute_ngram_mia_score(samples, synthetic)
        elif method == 'distance_str':
            # Call the function to ensure 'punkt' is downloaded
            download_punkt_if_not_exists()
            
            scores = compute_str_distance_mia_score(samples, synthetic, 
                                                distance_metrics=['jaccard', 'levenshtein', 'tfidf'], 
                                                ks = [1, 5, 10, 25])
        elif method == 'distance_emb':
            scores, synthetic_embeddings = compute_emb_distance_mia_score(samples, synthetic, synthetic_embeddings, 
                                                ks = [1, 5, 10, 25])
        all_mia_scores.update(scores)
        
    return all_mia_scores, synthetic_embeddings