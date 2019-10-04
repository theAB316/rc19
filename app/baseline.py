import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



	# constructing similarity matrix
def construct_sim_matrix(vectors):
	vectors = np.asarray(vectors)
	matrix = cosine_similarity(vectors)
	np.fill_diagonal(matrix, 0.0)
	return matrix


if __name__ == '__main__':
	pickle_in = open("vectors.pickle","rb")
	app_titles, vectors = pickle.load(pickle_in)

	matrix = construct_sim_matrix(vectors)

	data = pd.read_csv('googleplaystore.csv')

	selected_item = int(input("Enter a number from 0 to "+ str(len(vectors)) + " \n"))
	print(data.iloc[selected_item])
	max_val = max(matrix[selected_item])
	i, = np.where(matrix[selected_item] == max_val) # i return list
	rec_item = i[0]
	print(data.iloc[rec_item])
	

	


