
import nltk
from nltk.lm import Laplace
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from Levenshtein import ratio
import numpy as np
from tqdm import tqdm
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

VALID_METHODS = ['n-gram', 'sim_str', 'sim_emb']

### FOR SIMILARITY BASED METHODS
def jaccard_similarities(sample, synthetic_tokenized):
    # decide to consider tokens with capitalization and punctuation, as this is also how things are generated
    set1 = set(sample.split())
    return [1 - nltk.jaccard_distance(set1, set2) for set2 in synthetic_tokenized]

def levenshtein_similarities(sample, synthetic):
    return [ratio(sample.lower(), sentence2) for sentence2 in synthetic]

# def tfidf_distances(sample, tfidf_vectorizer, synthetic_tfidf):
    
#     # transform the sample too
#     tfidf_original = tfidf_vectorizer.transform([sample]).toarray()

#     # also get the binary version of the original tfidf
#     tfidf_original_binary = tfidf_original.copy()
#     tfidf_original_binary[tfidf_original_binary > 0] = 1
    
#     # we first want to compute the sum of the importance in the original sentence that is present in each synthetic sequence
#     similarities = tfidf_original @ synthetic_tfidf.T

#     # but we need to normalize this by how many words contribute to that, ie the union of the words in the original and synthetic
#     norm_factors = tfidf_original_binary @ synthetic_tfidf.T

#     return [1 - sim / norm_factors[0][j]  if norm_factors[0][j] > 0 else 1 for j, sim in enumerate(similarities[0])]

def compute_str_similarity_mia_score(samples, synthetic, 
                     sim_metrics=['jaccard', 'levenshtein'], 
                     ks = [1, 5, 10, 25]):
    sample_scores = {}

    # do some preprocessing on synthetic
    synthetic_tokenized = [set(synthetic_sample.split()) for synthetic_sample in synthetic]

    # fit and transform tfidf on synthetic data - only keeping (0,1) values
    # tfidf_vectorizer = TfidfVectorizer()
    # synthetic_tfidf = tfidf_vectorizer.fit_transform(synthetic).toarray()
    # synthetic_tfidf[synthetic_tfidf > 0] = 1

    for sim_metric in sim_metrics:
        for i, sample in tqdm(enumerate(samples), desc=f'Computing similarity MIA scores for {sim_metric}'):
            # if distance_metric == 'tfidf':
            #     distances = tfidf_distances(sample, tfidf_vectorizer, synthetic_tfidf)
            if sim_metric == 'jaccard':
                similarities = jaccard_similarities(sample, synthetic_tokenized)
            elif sim_metric == 'levenshtein':
                similarities = levenshtein_similarities(sample, synthetic)
            else:
                raise ValueError(f"Invalid similarity metric: {sim_metric}")
            similarities_sorted = np.sort(similarities)[::-1] # we need it from high to low
            for k in ks:
                mean_closest_k = np.mean(similarities_sorted[:k])
                if i == 0:
                    sample_scores[f'{sim_metric}_{k}'] = [mean_closest_k]
                else:
                    sample_scores[f'{sim_metric}_{k}'].append(mean_closest_k)

    return sample_scores

def compute_emb_sim_mia_score(samples, synthetic, synthetic_embeddings, ks = [1, 5, 10, 25], 
                                   emb_model_name = 'paraphrase-MiniLM-L6-v2'): # 'all-mpnet-base-v2':
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer(emb_model_name)

    if synthetic_embeddings is None:
        print("Computing the embeddings for the synthetic data")
        synthetic_embeddings = model.encode(synthetic)
    
    sample_scores = {}
    sample_embeddings = model.encode(samples)
    embedding_similarities = model.similarity(sample_embeddings, synthetic_embeddings).numpy()

    for i, sample in tqdm(enumerate(samples), desc=f'Computing distance MIA scores for embedding similarity'):
        similarities = embedding_similarities[i]
        # rescale the similarities to be in [0,1] (for now it can be [-1, 1])
        similarities = (similarities + 1) / 2
        similarities_sorted = np.sort(similarities)[::-1] # we need it from high to low
        for k in ks:
            mean_closest_k = np.mean(similarities_sorted[:k])
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
    tokens = word_tokenize(text)
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

def ngram_log_likelihood(lm, text):
    """
    Compute the log likelihood of the n-gram model on a given piece of text.
    The log likelihood is the sum of the log likelihood of the n-grams in the text.
    """
    ngrams = generate_ngrams(text, lm.order)
    log_likelihood = 0
    for ngram in ngrams:
        tokens = word_tokenize(ngram)
        prob = lm.score(tokens[lm.order-1], context=tokens[:lm.order-1])
        log_likelihood += np.log(prob).astype(np.double)

    return log_likelihood

def compute_ngram_mia_score(samples, synthetic, ns = [1, 2, 3, 4]):
    sample_scores = {}
    for n in ns:
        print(f"Training the {n}-gram model on and computing its losses.")
        tokenized_train_text = [word_tokenize(sentence) for sentence in synthetic]
        train_data, vocab_data = padded_everygram_pipeline(n, tokenized_train_text)
        model = Laplace(n)
        model.fit(train_data, vocab_data)
        sample_scores[f'ngram_{n}'] = [ngram_log_likelihood(model, sample) for sample in samples]

    return sample_scores

def compute_mia_score(samples, synthetic, method, synthetic_embeddings = None):
    all_mia_scores = {}
    if method == 'all':
        methods = VALID_METHODS
    for method in methods:
        if method == 'n-gram':
            scores = compute_ngram_mia_score(samples, synthetic)
        elif method == 'sim_str':            
            scores = compute_str_similarity_mia_score(samples, synthetic, 
                                                sim_metrics=['jaccard', 'levenshtein'], 
                                                ks = [1, 5, 10, 25])
        elif method == 'sim_emb':
            scores, synthetic_embeddings = compute_emb_sim_mia_score(samples, synthetic, synthetic_embeddings, 
                                                ks = [1, 5, 10, 25])
        all_mia_scores.update(scores)

    # MIA scores need to be in [0,1]
    for key in all_mia_scores:
        if 'ngram' not in key:
            try: 
                # adding a certain margin to the range to account for floating point errors
                assert all(0 - 1e-3 <= score <= 1 + 1e-3 for score in all_mia_scores[key]), "Some scores are out of the range [0, 1]"
            except AssertionError:
                for i in range(len(samples)):
                    print(samples[i], all_mia_scores[key][i])
                raise AssertionError
        
    return all_mia_scores, synthetic_embeddings