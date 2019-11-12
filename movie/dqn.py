import numpy as np
import pandas as pd
import pickle
import random
import collections 

from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K


batch_size = 10
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
output_dim = 53           ## this needs to be changed later???

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

def create_hash(items):
    global movies_watched
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

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
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

def select_action(state, policy_net, items):
    global movies_watched

    rate = eps_start    #this needs to be changed; decay needs to be added in!!!

    if random.random() > rate:
        #get action from q table/network
        action = np.argmax(policy_net.predict(state))     # here predict is used to get action
    else:
        action = random.randrange(0, len(items))

    movies_watched[items[action].id] = True     # sets hash table value to True for that particular movie
    return action


def get_reward(action, items):
    if movies_watched[items[action].id] == True:
        reward = 0                  #reward should be 0 so that movies arent repeated
    else:   
        reward = items[action].rating
    
    return reward

def get_state(state, action, items):
    #state = list(state)
    state = state[svd_vector_dim: ]
    state.extend(items[action].vector)

    return state

    


def main():
    path = 'data/'
    data_m, data_r = preproc(path)
    vectors = create_tfidf_svd(data_m['title_genre'], svd_vector_dim) 
    items = create_item_vectors(data_r, vectors)

    create_hash(items)
    
    policy_net = DQN(input_dim=state_stack_size*svd_vector_dim, output_dim=output_dim)
    target_net = DQN(input_dim=state_stack_size*svd_vector_dim, output_dim=output_dim)


    memory = Memory()
    for episode in range(1):#num_episodes):
        #reset
        state = get_initial_state(items)     ## this should get the initial state for each episode

        #timesteps = len(items)
        timesteps = 53
        for count, timestep in enumerate(range(timesteps)):
            action = select_action(state, policy_net, items)    #action is a number, indicating which movie id is selected
            reward = get_reward(action, items)
            next_state = get_state(state, action, items)       # passing old state
            memory.push((state, action, next_state, reward))

            
            
            X = []
            y = []
            if memory.size > batch_size:
                batch = memory.sample(batch_size)       # list of (s,a,n,r) 

                for i, _ in enumerate(batch):
                    state, action, next_state, reward = batch[i]

                    states = np.asarray([state])                # for syntax purposes (of fit method)
                    next_states = np.asarray([next_state])

                    current_q_vector = policy_net.predict(states)   # contains list of vectors. each vector(the Q values) is of length equal to number of movies rated by user. 53 for 1st user
                    future_q_vector = target_net.predict(next_states)

                    max_future_q = np.max(future_q_vector[0])
                    new_q = reward + gamma * max_future_q       # q-learning update rule

                    current_q_vector[0][action] = new_q         
                    # current_q_vector[0] gives one 53 dim vector. for that vector update particular q value
                    # for whatever action (movie id) you take, you updates its q value.


                    X.append(state)
                    y.append(current_q_vector[0])   # append the vector, NOT list(one vector)

                X = np.asarray(X)
                y = np.asarray(y)

                policy_net.fit(X, y, verbose=1)
            

            state = next_state

        if count%10 == 0:
            target_net.fit(X, y, verbose=1)




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