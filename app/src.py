import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


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



data = pd.read_csv('googleplaystore.csv')
print(list(data.columns))
#print(data.shape)

vectors = create_tfidf(data['App'])
svd = TruncatedSVD(n_components = 300)
svd = svd.fit_transform(vectors)    #ndarray











