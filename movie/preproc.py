import numpy as np
import pandas as pd




def main():
	path = 'data/'
	data_m = pd.read_csv(path+'movies.csv')
	data_r = pd.read_csv(path+'ratings.csv')
	data_u = pd.read_csv(path+'users.csv')

	movie_ratings = pd.merge(data_m, data_r)
	lens = pd.merge(movie_ratings, data_u)
	#print(lens.head())

	most_rated = lens.groupby('title').size().sort_values(ascending=False)[:25]
	print(most_rated)


if __name__ == '__main__':
	main()