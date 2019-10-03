import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 


def remove_stopwords(corpus):
    corpus_f = []
    stop = list(stopwords.words('english'))
    for word in corpus:
        if word not in stop:
            corpus_f.append(word)
    return corpus_f

def create_tfidf(app_titles):
    vectorizer = TfidfVectorizer(min_df=2, max_features=70000, strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w+', use_idf=True, 
                            smooth_idf=True, sublinear_tf=True, stop_words = 'english')
    vectors = vectorizer.fit_transform(app_titles)
    return vectors

def encode(data, feature):
    le = preprocessing.LabelEncoder()
    le.fit(list(data[feature].values))
    data[feature] = le.transform(list(data[feature]))


data = pd.read_csv('googleplaystore.csv')
print(list(data.columns))
#print(data.shape)

vectors = create_tfidf(data['App'])
svd = TruncatedSVD(n_components = 300)
vectors = svd.fit_transform(vectors)    #ndarray

data.drop(['App', 'Last Updated', 'Current Ver', 'Android Ver'], axis=1, inplace=True)

# for i in range(0, len()):
#     data[' = row[:-1]


for row in ['Category', 'Type', 'Content Rating', 'Genres']:
    #print(row)
    encode(data, row)



for index, row in data.iterrows():
    print(list(row))
    break
    #vectors.append(list(row))













