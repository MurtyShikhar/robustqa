import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import args as args_dependency
import dataset as ds
import string
from typing import List
import numpy as np
import json
import hashlib
import pickle
import os

from features.FeatureFunction import FeatureFunction
from features.AvgSentenceLen import AvgSentenceLen
from features.MaxWordRepetition import MaxWordRepetition
from features.NumberOfAlnums import NumberOfAlnums
from features.WordVariety import WordVariety
from features.NumberOfCommas import NumberOfCommas



# If we come up with feature extractors we should add them to this list
CUSTOM_FEATURE_EXTRACTORS: List[FeatureFunction] = [AvgSentenceLen(), MaxWordRepetition(), NumberOfCommas(), NumberOfAlnums(),
                                                    WordVariety()]


def extract_custom_features(contexts: List[str]):
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

def main():
    args = args_dependency.get_train_test_args()

    nltk.download('stopwords')
    nltk.download('wordnet')

    max_tfidf_features = 300
    custom_feature_scale = 6

    # read data
    data = ['datasets/indomain_train/squad',
	   'datasets/indomain_train/nat_questions',
	   'datasets/indomain_train/newsqa',
	   'datasets/oodomain_train/duorc',
	   'datasets/oodomain_train/race',
	   'datasets/oodomain_train/relation_extraction']

    all_data = {}
    for i in data:
        data_dict = ds.read_squad(i, 'save')
        all_data = ds.merge(data_dict, all_data)

    X_train = list(set(all_data['context']))

    # for use in constructing co-occurrence matrix
    text_to_id_dict = dict(zip(all_data['context'], all_data['topic_id']))

    # get custom features before modifying contexts
    custom_features = extract_custom_features(X_train)

    #Vectorisation : -
    cached_processed = 'save/all_train_text_processed3'
    if os.path.exists(cached_processed):
        X_train_processed = pickle.load(open(cached_processed, 'rb'))
    else:
        X_train_processed = [' '.join(text_process(item)) for item in X_train]
        pickle.dump(X_train_processed, open(cached_processed, 'wb'))

    tfidfconvert = TfidfVectorizer(max_features=max_tfidf_features, sublinear_tf=True, max_df=0.7, min_df=0.0001).fit(X_train_processed)

    X_transformed=tfidfconvert.transform(X_train_processed)
    pickle.dump(tfidfconvert, open("save/tfidf_max07_min00001_2.pickle", "wb"))
    pickle.dump(X_transformed, open("save/train_text_features_max07_min00001_2.pickle", "wb"))

    custom_features = normalize_matrix_so_cols_have_zero_mean_unit_variance(custom_features)
    custom_features *= 1/(max_tfidf_features ** 0.5) * custom_feature_scale

    # append the custom features for the full feature set
    raw_k_means_features = np.concatenate((X_transformed.toarray(), custom_features), axis=1)

    # normalize each column to have 0 mean and unit variance
    k_means_features = normalize(raw_k_means_features, axis=1)
    np.save('save/kmeansfeatures', k_means_features)

    # Cluster the training sentences with K-means technique
    km = KMeans(n_clusters=20, n_init=30)
    modelkmeans20 = km.fit(k_means_features)

    hist, bins = np.histogram(modelkmeans20.labels_, bins=20)
    print (hist)

    kmeans_dict = {get_hash_str(X_train[idx]): int(label) for idx, label in enumerate(modelkmeans20.labels_)}

    with open('save/topic_id_pair_kmeans20_300', 'w') as f:
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
    np.save(f'save/kmeans_co_occurance_20_300', co_occurance)

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
    main()
