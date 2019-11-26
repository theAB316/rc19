from dqn import preproc, create_tfidf_svd, create_item_vectors_all_users, get_initial_state
from dqn import svd_vector_dim, Item

import random
import numpy as np
import pickle
import pandas as pd

#from keras.models import load_model
def mergeDatasets(path):    #takes the path and returns a merged dataset
    data_m = pd.read_csv(path+'movies.csv')
    data_r = pd.read_csv(path+'ratings.csv')
    data_u = pd.read_csv(path+'users.csv')

    movie_ratings = pd.merge(data_m, data_r)
    lens = pd.merge(movie_ratings, data_u)
    #print(lens.head())

    #most_rated = lens.groupby('title').size().sort_values(ascending=False)[:25]
    #print(most_rated)

    return lens

def createUserItemPair(data):
    ## picks top k rated movies & top k users with most ratings
    ## selects subset containing only above users and movies
    ## creates (user, item) pair matrix with rating as cell value
    lens = data.copy()

    most_rated_users = lens.groupby('userId').size().sort_values(ascending=False)
    mru_keys = list(most_rated_users.keys())

    most_rated_movies = lens.groupby('itemId').size().sort_values(ascending=False)
    mrm_keys = list(most_rated_movies.keys())
    
    
    data = data.loc[data['userId'].isin(mru_keys) & data['itemId'].isin(mrm_keys)]
    data = data.pivot(index='userId', columns='itemId', values='rating')
    #print(data.head())
    return data 

def avg_rating_movies(path):
    data = mergeDatasets(path)
    data = createUserItemPair(data)
    
    data = data.transpose()
    data['mean'] = data.mean(axis=1, skipna=True)
    print(data.sample(10))



def main():
    path = 'data/'
    avg_rating_movies(path)

    exit(0)

    with open("out_files/items.pickle", "rb") as f:
        items = pickle.load(f)

    with open("out_files/target_net.pickle", "rb") as f:
        target_net = pickle.load(f)



    #target_net = load_model('out_files/target_net.h5')   

    random_users_id = random.sample(range(100, 6040), 20)
        
    output = []
    for user in random_users_id:
        state = get_initial_state(items[user])

        states = np.asarray([state])
        q_vector = target_net.predict(states)
        action = np.argmax(q_vector)

        print(np.max(q_vector), np.min(q_vector))


        output.append(action)

    print("\n\n\n", output)



if __name__ == '__main__':
    main()