#Text pre-processing
"""removes punctuation, stopwords, and returns a list of the remaining words, or tokens"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import args
import util

args = args.get_train_test_args()

nltk.download('stopwords')
nltk.download('wordnet')

#Cleaning the text

import string
def text_process(text):
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

# read data
data = util.read_squad("datasets_augmented/indomain_train/squad_subset", args['save_dir'])
X_train = data['context']
#print (X_train)

#Vectorisation : -
tfidfconvert = TfidfVectorizer(analyzer=text_process).fit(X_train)

X_transformed=tfidfconvert.transform(X_train)
X_transformed_array = X_transformed.toarray()
normalized_X_transformed = normalize(X_transformed_array, axis=1)
# Clustering the training sentences with K-means technique

K = range(1,100)
Sum_of_squared_distances = []
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(normalized_X_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
