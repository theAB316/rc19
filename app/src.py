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

def remove_plus(t):
    t = t[:-1]
    #print(t)
    return t


data = pd.read_csv('googleplaystore.csv')
#print(data.shape)

vectors = create_tfidf(data['App'])
svd = TruncatedSVD(n_components = 300)
vectors = svd.fit_transform(vectors)    #ndarray
vectors = vectors.tolist()


data.drop(['App', 'Last Updated', 'Current Ver', 'Android Ver'], axis=1, inplace=True)
print(list(data.columns))

for i in range(0, len(data['Installs'])):
    data['Installs'][i] = data['Installs'][i].replace(',', '')
    data['Installs'][i] = int(data['Installs'][i].replace('+', ''))

for i in range(0, len(data['Size'])):
    if data['Size'][i][-1] == 'k':
        data['Size'][i] = float(data['Size'][i].replace('k', ''))

    elif data['Size'][i][-1] == 'M':
        data['Size'][i] = data['Size'][i].replace('M', '')
        data['Size'][i] = float(data['Size'][i])*1000.0

    else:
        data['Size'][i] = float(data['Size'][i].replace('Varies with device', '18152'))

for i in range(0, len(data['Price'])):
    data['Price'][i] = float(data['Price'][i].replace('$', ''))

    

for row in ['Category', 'Type', 'Content Rating', 'Genres']:
    encode(data, row)

print(vectors.shape)
for index, row in data.iterrows():
    vectors[index].extend(list(row))
print(vectors.shape)
















