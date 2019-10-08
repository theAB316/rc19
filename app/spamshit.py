#we stopped working on app

data1 = pd.read_csv('googleplaystore.csv')
print(len(set(data1['App'])))

#this gives 9659, but user_reviews.csv gives 64295


















from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


vectorizer = TfidfVectorizer(min_df=2, max_features=300, strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w+', use_idf=True, 
                            smooth_idf=True, sublinear_tf=True, stop_words = 'english')

vectors = vectorizer.fit_transform(data['App'])


'''feat = ['Genre'] 
for x in feat: 
	le = LabelEncoder() 
	le.fit(list(genre[x].values)) 
	genre[x] = le.transform(list(genre[x]))'''


'''corpus = set()
for title in data['App']:
    for word in title.split(" "):
        corpus.add(word.lower())

corpus = remove_stopwords(corpus)'''


def create_doc2vec(app_titles):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print("Model Saved")
    