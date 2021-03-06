import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import args as args_dependency
import dataset
import string
from typing import List
import numpy as np

from features.FeatureFunction import FeatureFunction
from features.AvgSentenceLen import AvgSentenceLen


# If we come up with feature extractors we should add them to this list
CUSTOM_FEATURE_EXTRACTORS: List[FeatureFunction] = [AvgSentenceLen()]


def extract_custom_features(contexts: List[str]):
    # TODO: if this gets too slow, we can cache this in a file
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

def normalize_matrix_so_cols_have_zero_mean_unit_variance(mtx: np.ndarray) -> np.ndarray:
    mtx -= np.mean(mtx, axis=0).reshape(1, -1)
    mtx /= np.std(mtx, axis=0).reshape(1, -1)
    return mtx

def main():
    args = args_dependency.get_train_test_args()

    nltk.download('stopwords')
    nltk.download('wordnet')

    # read data
    # TODO: FIX BEFORE COMMIT
    data = dataset.read_squad("datasets_augmented/indomain_train/squad_subset", args['save_dir'])
    X_train = data['context']

    # get custom features before modifying contexts
    custom_features = extract_custom_features(X_train)

    #Vectorisation : -
    tfidfconvert = TfidfVectorizer(analyzer=text_process).fit(X_train)

    X_transformed=tfidfconvert.transform(X_train)
    X_transformed_array = X_transformed.toarray()

    # append the custom features for the full feature set
    raw_k_means_features = np.concatenate((X_transformed_array, custom_features), axis=1)

    # normalize each column to have 0 mean and unit variance
    k_means_features = normalize_matrix_so_cols_have_zero_mean_unit_variance(raw_k_means_features)

    # Cluster the training sentences with K-means technique
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


if __name__ == "__main__":
    main()
