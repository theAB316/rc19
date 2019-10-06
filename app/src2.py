import numpy as np
import pandas as pd




def main():
	data = pd.read_csv('googleplaystore_user_reviews.csv')
	#print(data.groupby('App').count())
	for index, row in data.iterrows():
		print(row)
		if index > 1:
			break


if __name__ == '__main__':
	main()