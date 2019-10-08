import numpy as np
import pandas as pd


def mergeDatasets(path):	#takes the path and returns a merged dataset
	data_m = pd.read_csv(path+'movies.csv')
	data_r = pd.read_csv(path+'ratings.csv')
	data_u = pd.read_csv(path+'users.csv')

	movie_ratings = pd.merge(data_m, data_r)
	lens = pd.merge(movie_ratings, data_u)
	#print(lens.head())

	#most_rated = lens.groupby('title').size().sort_values(ascending=False)[:25]
	#print(most_rated)

	return lens

def createUserItemPair(data, n_users, n_items):
	## picks top k rated movies & top k users with most ratings
	## selects subset containing only above users and movies
	## creates (user, item) pair matrix with rating as cell value
	lens = data.copy()

	most_rated_users = lens.groupby('userId').size().sort_values(ascending=False)[:n_users]
	mru_keys = list(most_rated_users.keys())

	most_rated_movies = lens.groupby('itemId').size().sort_values(ascending=False)[:n_items]
	mrm_keys = list(most_rated_movies.keys())
	
	
	data = data.loc[data['userId'].isin(mru_keys) & data['itemId'].isin(mrm_keys)]
	data = data.pivot(index='userId', columns='itemId', values='rating')
	#print(data.head())
	return data	




def main():
	path = 'data/'
	data = mergeDatasets(path)

	data = createUserItemPair(data, n_users=200, n_items=400)
	print(data.sample(5))




if __name__ == '__main__':
	main()