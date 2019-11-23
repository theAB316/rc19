import numpy as np
import pandas as pd
import pickle
import math
import random
import collections 
from time import time
from tqdm import tqdm

from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.models import load_model

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard


batch_size = 10
gamma = 0.999

eps_start = 1
eps_end = 0.01
eps_decay = 0.001

target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 100        # number of users = 6040


svd_vector_dim = 300       ## vector dim for svd
state_stack_size = 4       ## this is the no. of consecutive movie vectors used for state 
output_dim = 2314           ## max number of movies a user has watched

movies_watched = dict()

class Item():
    ## item object is 3-tuple
    def __init__(self, id, vector, rating):
        self.id = id
        self.vector = vector
        self.rating = rating

class Memory():
    def __init__(self):
        self.size = 0
        self.memory = collections.deque([])

    def push(self, experience):
        global memory_size

        if self.size >= memory_size:
            self.memory.popleft()
            self.memory.append(experience)
        else:
            self.memory.append(experience)
            self.size += 1

    def sample(self, batch_size):
        batch = random.sample(list(self.memory), batch_size)
        return batch

    def clear(self):
        self.size = 0
        self.memory.clear()




def preproc(path):
    ## creates a new column in data_m - title_genre that has a concatenation of title & genre
    data_r = pd.read_csv(path+'ratings.csv')
    data_m = pd.read_csv(path+'movies.csv')

    #print(len(np.unique(data_r['userId'])))
    
    
    
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


def create_item_vectors(data_r, vectors, userId):
    ## this returns a list of movie vectors watched by each user
    ## currently only for 1st user

    item_ids = list(data_r.loc[data_r['userId'] == userId]['itemId']) 


    ratings = list(data_r.loc[data_r['userId'] == userId]['rating']) 
    
    items = []

    
    for i in range(len(item_ids)):
        if np.isnan(ratings[i]):
            pass

        items.append(Item(item_ids[i], vectors[item_ids[i]], ratings[i]))

    return items #item objects for each user

def create_item_vectors_all_users(data_r, vectors):
    items = []
    
    for i in range(num_episodes):   #no. of users
        items.append(create_item_vectors(data_r, vectors, i))
    
    return items        #len should be 6040



def create_hash(items):
    # hash of movies watched/not
    global movies_watched

    movies_watched.clear()        # empty dictionary for each user/episode
    for item in items:
        movies_watched[item.id] = False

    return 1
    


def DQN(input_dim, output_dim, action=None):
    ## creates the DQN model, needs paramater tuning

    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='mse', optimizer='adam')
    ## action - a
    return model


def get_initial_state(items):
    ## concatenates the first k(=state_stack_size) movie vectors. which is the initial state


    state = []

    for item in items[:state_stack_size]:
        state.extend(item.vector)



    # now we remove those movies from the item_vector list so that it isnt going to be recommended again
    # here we are deleting from the original item_vectors; bcuz in python references are passed 
    #del items[:state_stack_size]

    return state

def select_action(state, policy_net, items, timestep):
    
    #this needs to be changed; decay needs to be added in!!!
    rate = eps_end + (eps_start - eps_end) * (math.exp(-1 * timestep * eps_decay))

    if random.random() > rate:
        #get action from q table/network
        states = np.asarray([state])
        action = np.argmax(policy_net.predict(states))     # here predict is used to get action
    else:
        action = random.randrange(0, len(items))

    return action


def get_reward(action, items, data_r):
    # data_r is required to calc avg rating for a particular movie
    # (already_watched, going_to_watch, never_watched)
    global movies_watched

    if (action in movies_watched) and (movies_watched[action] == True): # already_watched case
        reward = 0                  #reward should be 0 so that movies arent repeated
    else:
        if(action not in movies_watched):
            # reward is the avg ratings for the movie by all users
            # never_watched case
            reward = np.mean(list(data_r.loc[data_r['itemId'] == action]['rating']))
        
        else:
            # going_to_watch
            print(action, items[action].id)
            reward = items[action].rating
            movies_watched[items[action].id] = True     # sets hash table value to True for that particular movie


    
    return reward



def get_state(state, action, items, vectors):
    # vectors is reqd, to get the vector of the movie not watched by user at all
    #state = list(state)
    state = state[svd_vector_dim: ]

    if(action not in items):
        state.extend(vectors[action])
    else:    
        state.extend(items[action].vector)

    return state

    


def main():
    path = 'data/'
    data_m, data_r = preproc(path)
    vectors = create_tfidf_svd(data_m['title_genre'], svd_vector_dim) 
    items = create_item_vectors_all_users(data_r, vectors)

        # with open("out_files/items.pickle", "wb") as f:
        #     pickle.dump(items, f)

    
    policy_net = DQN(input_dim=state_stack_size*svd_vector_dim, output_dim=output_dim)
    target_net = DQN(input_dim=state_stack_size*svd_vector_dim, output_dim=output_dim)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    memory = Memory()
    for episode in tqdm(range(num_episodes)):
        memory.clear()                       # reset
        create_hash(items[episode])

        state = get_initial_state(items[episode])     ## this should get the initial state for each episode

        total_reward = 0

        timesteps = len(items[episode])     # timesteps is no. of movies watched by each user
        for count, timestep in enumerate(range(timesteps)):
            action = select_action(state, policy_net, items[episode], timestep)    #action is a number, indicating which movie id is selected
            reward = get_reward(action, items[episode], data_r)
            next_state = get_state(state, action, items[episode], vectors)       # passing old state
            memory.push((state, action, next_state, reward))
            
            
            X = []
            y = []
            if memory.size > batch_size:
                batch = memory.sample(batch_size)       # list of (s,a,n,r) 

                for i, _ in tqdm(enumerate(batch)):
                    state, action, next_state, reward = batch[i]
                    total_reward += reward
                    #print(action, reward)

                    states = np.asarray([state])                # for syntax purposes (of fit method)
                    next_states = np.asarray([next_state])

                    current_q_vector = policy_net.predict(states)   # contains list of vectors. each vector(the Q values) is of length equal to number of movies rated by user. 53 for 1st user
                    future_q_vector = target_net.predict(next_states)

                    max_future_q = np.max(future_q_vector[0])
                    new_q = reward + gamma * max_future_q       # q-learning update rule

                    current_q_vector[0][action] = new_q   
                    # because list of one vector      
                    # for that vector update particular q value
                    # for whatever action (movie id) you take, you updates its q value.


                    X.append(state)
                    y.append(current_q_vector[0])   # append the vector, NOT list(one vector)

                X = np.asarray(X)
                y = np.asarray(y)

                with tf.device('/gpu:0'):
                    policy_net.fit(X, y, verbose=0)

                with open("out_files/policy_net.pickle", "wb") as f:
                    pickle.dump(policy_net, f)
            

            state = next_state

        if count%1 == 0:
            with tf.device('/gpu:0'):
                target_net.fit(X, y, verbose=0)# callbacks=[tensorboard])

            with open("out_files/target_net.pickle", "wb") as f:
                pickle.dump(target_net, f)
            #target_net.save('out_files/target_net.h5')

        print(total_reward)




    # s = []
    # for i in range(4):
    #     s.extend(list(items[i].vector))
    # for i in range(4, len(items)):
    #     predict_s = np.asarray([s]) 
    #     output = target_net.predict(predict_s)
    #     s = s[svd_vector_dim:] + list(items[i].vector)

    #     print(output)


            



        




if __name__ == '__main__':
    main()