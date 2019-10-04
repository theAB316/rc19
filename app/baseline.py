import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



	# constructing similarity matrix
def construct_sim_matrix(vectors):
	vectors = np.asarray(vectors)
	matrix = cosine_similarity(vectors)
	np.fill_diagonal(matrix, 0.0)
	return matrix


if __name__ == '__main__':
	pickle_in = open("vectors.pickle","rb")
	vectors = pickle.load(pickle_in)

	matrix = construct_sim_matrix(vectors)

	selected_item = int(input("Enter a number from 0 to "+ str(len(vectors)) + " \n"))
	max_val = max(matrix[selected_item])
	i, = np.where(matrix[selected_item] == max_val) # i return list
	print(i)
	rec_item = vectors[i[0]]
	print(len(rec_item))
	print(matrix)

	


