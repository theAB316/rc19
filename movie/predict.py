from dqn import preproc, create_tfidf_svd, create_item_vectors_all_users, get_initial_state
from dqn import svd_vector_dim, Item

import random
import numpy as np
import pickle

#from keras.models import load_model

def main():
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

        print(np.max(q_vector), end=" ")

        output.append(action)

    print("\n\n\n", output)



if __name__ == '__main__':
    main()