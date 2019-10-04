import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


pickle_in = open("vectors.pickle","rb")
vectors = pickle.load(pickle_in)

print(len(vectors), len(vectors[0]))

# constructing similarity matrix
print(type(vectors))

np.replace()
#matrix = cosine_similarity(vectors)
#print(len(matrix), len(matrix[0]))
#print(matrix)