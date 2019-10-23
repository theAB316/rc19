import numpy as np
import pandas as pd
import pickle
import random

from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K


batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000


svd_vector_dim = 300       ## vector dim for svd
state_stack_size = 4       ## this is the no. of consecutive movie vectors used for state 
output_dim = 20           ## this needs to be changed later???



class Item():
    ## item object is 3-tuple
    def __init__(self, id, vector, rating):
        self.id = id
        self.vector = vector
        self.rating = rating

def preproc(path):
    ## creates a new column in data_m - title_genre that has a concatenation of title & genre
    data_r = pd.read_csv(path+'ratings.csv')
    data_m = pd.read_csv(path+'movies.csv')
    
    
    data_m['title_genre'] = data_m['title'] + data_m['genre'] #choice b/w doing separately or together
    data_m.drop([ 'origin_iid', 'title', 'genre'], axis='columns', inplace=True)
    
    return data_m, data_r


def create_tfidf_svd(str_list, svd_vector_dim):
    ## str_list is list of title_genre 

    vectorizer = TfidfVectorizer(min_df=2, max_features=70000, strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w+', use_idf=True, 
                            smooth_idf=True, sublinear_tf=True, stop_words = 'english')
    vectors = vectorizer.fit_transform(str_list)
    svd = TruncatedSVD(n_components = svd_vector_dim)
    vectors = svd.fit_transform(vectors)    #ndarray
    #vectors = vectors.tolist()
    return vectors


def create_item_vectors(data_r, vectors):
    ## this returns a list of movie vectors watched by each user
    ## currently only for 1st user

    item_ids = list(data_r.loc[data_r['userId'] == 0]['itemId']) #selecting first user's item ids
    ratings = list(data_r.loc[data_r['userId'] == 0]['rating']) #selecting first user's ratings
    
    items = []
    
    for id in item_ids:
        if np.isnan(ratings[id]):
            pass

        items.append(Item(item_ids[id], vectors[id], ratings[id]))

    return items



def DQN(input_dim, output_dim, action=None):
    ## creates the DQN model, needs paramater tuning

    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ## action - a
    return model


def get_initial_state(items):
    ## concatenates the first k(=state_stack_size) movie vectors. which is the initial state

    state = np.array([])
    for item in items[:state_stack_size]:
        state = np.concatenate([state, item.vector])

    # now we remove those movies from the item_vector list so that it isnt going to be recommended again
    # here we are deleting from the original item_vectors; bcuz in python references are passed 
    del items[:state_stack_size]

    return state

def select_action(state, policy_net, items):
    rate = eps_start    #this needs to be changed; decay needs to be added in!!!

    if random.random() > rate:
        #get action from q table/network
    else:
        return random.randrange(0, len(items))

    


def main():
    path = 'data/'
    data_m, data_r = preproc(path)
    vectors = create_tfidf_svd(data_m['title_genre'], svd_vector_dim) 
    items = create_item_vectors(data_r, vectors)
    print("len of items: ", len(items))



    
    policy_net = DQN(input_dim=state_stack_size*svd_vector_dim, output_dim=output_dim)
    target_net = DQN(input_dim=state_stack_size*svd_vector_dim, output_dim=output_dim)



    for episode in range(1):#num_episodes):
        state = get_initial_state(items)     ## this should get the initial state for each episode

        timesteps = len(items)
        for timestep in range(timesteps):
            action = select_action(state, policy_net)
            reward = take_action(action)
            next_state = get_state()
            memory.push((state, action, next_state, reward))
        




if __name__ == '__main__':
    main()