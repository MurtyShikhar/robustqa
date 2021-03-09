import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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
CUSTOM_FEATURE_EXTRACTORS: List[FeatureFunction] = [AvgSentenceLen(), MaxSentenceLen(), MinSentenceLen()
                                                    , AdjectivePercentage(), CoordinatingConjunctionPercentage()
                                                    , NounPercentage(), PrepositionPercentage()
                                                    , MaxWordRepetition(), NumberOfAlnums(), NumberOfCommas()
                                                    , SentimentAnalysis(), WordVariety()]


def extract_custom_features(log, contexts: List[str]):
    log.info(f'Extracting custom features...')
    cached_custom_features = 'clustering/all_custom_features'
    if os.path.exists(cached_custom_features):
        log.info("Loading custom features from cache...")
        return pickle.load(open(cached_custom_features, 'rb'))
    else:
        log.info("Extracting custom features...")
        custom_features = np.zeros((len(contexts), len(CUSTOM_FEATURE_EXTRACTORS)))
        for i in range(len(contexts)):
            if i % 100 == 0:
                log.info(f'Iteration {i}/{len(contexts)}')
            for j in range(len(CUSTOM_FEATURE_EXTRACTORS)):
                value = CUSTOM_FEATURE_EXTRACTORS[j].evaluate(contexts[i])
                custom_features[i, j] = value

        log.info("Saving custom features in cache...")
        pickle.dump(custom_features, open(cached_custom_features, 'wb'))
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
    # data =['datasets/indomain_train/newsqa_subset']
    data = ['datasets/indomain_train/squad', 'datasets/indomain_train/nat_questions', 'datasets/indomain_train/newsqa'
        ,'datasets/oodomain_train/duorc', 'datasets/oodomain_train/race', 'datasets/oodomain_train/relation_extraction']

    all_data = {}
    for i in data:
        log.info(f'Loading {i}...')
        data_dict = ds.read_squad(i, 'save')
        all_data = ds.merge(data_dict, all_data)

    return list(set(all_data['context'])), dict(zip(all_data['context'], all_data['topic_id']))

def read_text_from_cache(log, X_train):
    cached_processed = 'clustering/all_train_text_processed'
    if os.path.exists(cached_processed):
        log.info("Loading processed data from cache...")
        return pickle.load(open(cached_processed, 'rb'))
    else:
        log.info("Saving processed data in cache...")
        X_train_processed = [' '.join(text_process(item)) for item in X_train]
        pickle.dump(X_train_processed, open(cached_processed, 'wb'))

        return X_train_processed

def load_text(log):
    X_train, text_to_id_dict = get_contexts(log)
    # get custom features before modifying contexts
    custom_features = extract_custom_features(log, X_train)
    log.info("Normalizing custom features...")
    custom_features = normalize_matrix_so_cols_have_zero_mean_unit_variance(custom_features)
    X_train_processed = read_text_from_cache(log, X_train)

    return X_train, X_train_processed, custom_features, text_to_id_dict

def prepare_features(log, results_folder, max_tfidf_features, custom_feature_scale, X_train, custom_features):
    log.info(f"Scaling custom features with scale {custom_feature_scale}...")
    custom_features *= 1 / (max_tfidf_features ** 0.5) * custom_feature_scale

    log.info(f"Extracting TF/IDF features with max {max_tfidf_features}...")
    tfidfconvert = TfidfVectorizer(max_features=max_tfidf_features, sublinear_tf=True, max_df=0.7, min_df=0.0001).fit(X_train)
    X_transformed = tfidfconvert.transform(X_train)
    pickle.dump(tfidfconvert, open(f"clustering/tfidf_max07_min00001_2.pickle", "wb"))
    pickle.dump(X_transformed, open("clustering/train_text_features_max07_min00001_2.pickle", "wb"))

    # append the custom features for the full feature set
    raw_k_means_features = np.concatenate((X_transformed.toarray(), custom_features), axis=1)

    log.info("Normalizing concatenated features...")
    # normalize each column to have 0 mean and unit variance
    k_means_features = normalize(raw_k_means_features, axis=1)
    np.savetxt(f'{results_folder}/kmeansfeatures.csv', k_means_features, delimiter=',')

    return k_means_features

def cluster(log, results_folder, num_clusters, num_iters, k_means_features):
    # Cluster the training sentences with K-means technique
    log.info(f'Generating {num_clusters} clusters with kmeans...')
    km = KMeans(n_clusters=num_clusters, n_init=num_iters)
    clusters = km.fit(k_means_features)

    hist, bins = np.histogram(clusters.labels_, bins=num_clusters)
    log.info(f'Kmeans is complete. Histogram: {hist}')

    kmeans_dict = {get_hash_str(X_train[idx]): int(label) for idx, label in enumerate(clusters.labels_)}

    log.info(f"Saving kmeans clusters in {results_folder}/kmeans_clusters.json...")
    with open(f'{results_folder}/kmeans_clusters.json', 'w') as f:
        json.dump(kmeans_dict, f, indent=2)

    return clusters

def gen_cooccurrance_matrix(results_folder, text_to_id_dict, num_clusters, clusters, X_train):

    # Build the matrix with cluster IDs as rows, topic IDs as columns
    topics_id = []
    for k, v in text_to_id_dict.items():
        if str(v) not in topics_id:
            topics_id.append(str(v))
    num_topics = len(topics_id)
    co_occurance = np.zeros((num_clusters, num_topics), dtype=int)
    for idx, cluster in enumerate(clusters.labels_):
        topic_id = int(text_to_id_dict[X_train[idx]])
        co_occurance[int(cluster)][topic_id] += 1
    np.savetxt(f'{results_folder}/kmeans_co_occurance.csv', co_occurance
        , delimiter=',', header=','.join(topics_id), fmt="%d")

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

    if not os.path.exists("clustering"):
        os.makedirs("clustering")
    log = get_logger("clustering", "log_clustering")

    max_tf_idf_features = [100, 200, 300]
    custom_feature_scale = [2, 4, 6, 8, 10]
    num_clusters = [20, 30, 40, 50, 60, 70]
    num_iters = [300, 350, 400]

    results_folder_format = 'clustering/max_tfidf_{0}_custom_scale_{1}_num_clusters_{2}_iters_{3}'
    X_train, X_train_processed, custom_features, text_to_id_dict = load_text(log)
    for max_features in max_tf_idf_features:
        for scale in custom_feature_scale:
            for clusters in num_clusters:
                for iters in num_iters:
                    results_folder = results_folder_format.format(max_features, scale, clusters, iters)
                    if not os.path.exists(results_folder):
                        os.mkdir(results_folder)

                    k_means_features = prepare_features(log, results_folder, max_features, scale, X_train_processed, custom_features)
                    k_means_clusters = cluster(log, results_folder, clusters, iters, k_means_features)
                    gen_cooccurrance_matrix(results_folder, text_to_id_dict, clusters, k_means_clusters, X_train)
                    log.info("Trial complete...")

