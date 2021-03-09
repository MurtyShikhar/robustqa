import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import dataset as ds
import string
from typing import List
import numpy as np
import json
import hashlib
import pickle
import os

from util import get_logger
from args import get_train_test_args

from features.FeatureFunction import FeatureFunction
from features.AdjectivePercentage import AdjectivePercentage
from features.AvgSentenceLen import AvgSentenceLen
from features.CoordinatingConjunctionPercentage import CoordinatingConjunctionPercentage
from features.LanguageCount import LanguageCount
from features.MaxSentenceLen import MaxSentenceLen
from features.MaxWordRepetition import MaxWordRepetition
from features.MinSentenceLen import MinSentenceLen
from features.NounPercentage import NounPercentage
from features.NumberOfAlnums import NumberOfAlnums
from features.NumberOfCommas import NumberOfCommas
from features.PrepositionPercentage import PrepositionPercentage
from features.SentimentAnalysis import SentimentAnalysis
from features.WordVariety import WordVariety

# If we come up with feature extractors we should add them to this list
CUSTOM_FEATURE_EXTRACTORS: List[FeatureFunction] = [AdjectivePercentage(), AvgSentenceLen(), CoordinatingConjunctionPercentage()
                                                    , LanguageCount(), MaxSentenceLen(), MaxWordRepetition(), MinSentenceLen()
                                                    , NounPercentage(), NumberOfAlnums(), NumberOfCommas(), PrepositionPercentage()
                                                    , SentimentAnalysis(), WordVariety()]


def extract_custom_features(log, contexts: List[str]):
    log.info(f'Extracting custom features {i}...')
    custom_features = np.zeros((len(contexts), len(CUSTOM_FEATURE_EXTRACTORS)))
    for i in range(len(contexts)):
        for j in range(len(CUSTOM_FEATURE_EXTRACTORS)):
            custom_features[i, j] = CUSTOM_FEATURE_EXTRACTORS[j].evaluate(contexts[i])
    return custom_features

#Text pre-processing
def text_process(text):
    """removes punctuation, stopwords, and returns a list of the remaining words, or tokens"""
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words
    '''
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
    return [stemmer.lemmatize(word) for word in nopunc]

def get_hash_str(text):
    md_object = hashlib.md5(text.encode())
    return md_object.hexdigest()

def normalize_matrix_so_cols_have_zero_mean_unit_variance(mtx: np.ndarray) -> np.ndarray:
    mtx -= np.mean(mtx, axis=0).reshape(1, -1)
    mtx /= np.std(mtx, axis=0).reshape(1, -1)
    return mtx

def get_contexts(log):
    # read data
    data = ['datasets/indomain_train/squad', 'datasets/indomain_train/nat_questions', 'datasets/indomain_train/newsqa'
        ,'datasets/oodomain_train/duorc', 'datasets/oodomain_train/race', 'datasets/oodomain_train/relation_extraction']

    all_data = {}
    for i in data:
        log.info(f'Loading {i}...')
        data_dict = ds.read_squad(i, 'save')
        all_data = ds.merge(data_dict, all_data)

    return list(set(all_data['context'])), dict(zip(all_data['context'], all_data['topic_id']))

def read_from_cache(log):
    cached_processed = 'clustering/all_train_text_processed_brynnemh'
    if os.path.exists(cached_processed):
        log.info("Loading processsed data from cache...")
        return pickle.load(open(cached_processed, 'rb'))
    else:
        log.info("Saving processsed data in cache...")
        X_train_processed = [' '.join(text_process(item)) for item in X_train]
        pickle.dump(X_train_processed, open(cached_processed, 'wb'))

        return X_train_processed

def main(args):
    max_tfidf_features = args["max_tfidf_features"]
    custom_feature_scale = args["custom_feature_scale"]
    num_clusters = args["num_clusters"]
    num_iters = args["kmeans_iters"]

    results_folder = f'clustering/max_tfidf_{max_tfidf_features}_custom_scale_{custom_feature_scale}_num_clusters_{num_clusters}_iters_{kmeans_iters}'
    os.mkdir(results_folder)
    log = util.get_logger(results_folder, 'log_clustering')

    X_train, text_to_id_dict = get_contexts(log)

    # get custom features before modifying contexts
    custom_features = extract_custom_features(log, X_train)

    #Vectorisation : -
    X_train_processed = read_from_cache(log)
    log.info("Extracting TF/IDF features...")
    tfidfconvert = TfidfVectorizer(max_features=max_tfidf_features, sublinear_tf=True, max_df=0.7, min_df=0.0001).fit(X_train_processed)
    X_transformed=tfidfconvert.transform(X_train_processed)
    pickle.dump(tfidfconvert, open(f"clustering/tfidf_max07_min00001_2.pickle", "wb"))
    pickle.dump(X_transformed, open("clustering/train_text_features_max07_min00001_2.pickle", "wb"))

    log.info("Normalizing custom features...")
    custom_features = normalize_matrix_so_cols_have_zero_mean_unit_variance(custom_features)
    custom_features *= 1 / (max_tfidf_features ** 0.5) * custom_feature_scale

    # append the custom features for the full feature set
    raw_k_means_features = np.concatenate((X_transformed.toarray(), custom_features), axis=1)

    log.info("Normalizing custom features...")
    # normalize each column to have 0 mean and unit variance
    k_means_features = normalize(raw_k_means_features, axis=1)
    np.save(f'{results_folder}/kmeansfeatures', k_means_features)

    # Cluster the training sentences with K-means technique
    log.info("Generating clusters with kmeans...")
    km = KMeans(n_clusters=num_clusters, n_init=num_iters)
    modelkmeans20 = km.fit(k_means_features)

    hist, bins = np.histogram(modelkmeans20.labels_, bins=num_clusters)
    log.info(f'Kmeans is complete. Histogram: {hist}')

    kmeans_dict = {get_hash_str(X_train[idx]): int(label) for idx, label in enumerate(modelkmeans20.labels_)}

    log.info("Saving kmeans clusters...")
    with open(f'{results_folder}/topic_id_pair_kmeans_{num_clusters}_{num_iters}', 'w') as f:
        json.dump(kmeans_dict, f)

    # Build the matrix with cluster IDs as rows, topic IDs as columns
    topics_id = []
    for k, v in text_to_id_dict.items():
        if v not in topics_id:
            topics_id.append(v)
    num_topics = len(topics_id)
    co_occurance = np.zeros((num_topics, 20))
    for idx, cluster in enumerate(modelkmeans20.labels_):
        topic_id = int(text_to_id_dict[X_train[idx]])
        co_occurance[topic_id][int(cluster)] += 1
    np.save(f'{results_folder}/kmeans_co_occurance_{num_clusters}_{num_iters}', co_occurance)

    '''
    K = range(4,100)
    Sum_of_squared_distances = []
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(k_means_features)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    '''

if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

    args = get_train_test_args()
    main(args)
