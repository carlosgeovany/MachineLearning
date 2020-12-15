import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import ndcg_score


## Avoid zero division warning
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class MatrixFactorization():
	"""
	MatrizFactorization class constructor
	...

	Attributes
	----------

	:ratings:	pandas dataframe
		contains the ratings
	:data:	numpy array
		ratings df as a numpy array

	"""

	def __init__(self,ratings,movies_data):
		self.ratings = ratings
		self.data = self.ratings.to_numpy()
		self.movies_data = movies_data


	def fit(self, K=2, epochs=2, alpha=0.001):
		"""
		randomly initialize user/item factors from a Gaussian
		and calculates  U and V
		"""
		users = self.data.shape[0]
		movies = self.data.shape[1]
		U = np.random.normal(0,.1,(users,K))
		V = np.random.normal(0,.1,(movies,K))

		for epoch in range(epochs):
			for u in range(users):
				for m in range (movies):
					Y = self.data[u][m] - np.dot(U[u],V[m])
					temp = U[u,:]
					U[u,:] +=  alpha * Y * V[m]
					V[m,:] +=  alpha * Y * temp

		self.U = U
		self.V = V


	@property
	def matrix(self):
		"""
		:return: float
			dot product U*V
		"""
		matrix = np.dot(self.U, self.V.T)
		matrix[matrix < 0] = 0
		return matrix

	@property
	def score(self):
		"""
		:return: float
			NDCG for matrix product U*V using sckikit learn method
		"""
		return ndcg_score(self.ratings, self.matrix)



	def estimate(self, u, m):
		"""
		auxiliar function to get predictions
		"""
		u,m = int(u), int(m)
		return np.dot(self.U[3],self.V[3]).round()


	def top_recommends(self, user, top=5):
		"""
		movies recommended for a user

		:user:	int
			a single user to recommend
		:top:	int
			max number of movies to be recommended
		:return:	pandas df
			df with movies recommended for a user
		"""
		## Reformat full matrix
		pred_matrix = (pd.DataFrame(self.matrix, columns = self.ratings.columns, index = self.ratings.index).
	               unstack().
	               reset_index(name='rating').
	               set_index('userId').sort_index(axis = 0).
	               reset_index().
	               rename(columns={'rating':'Must watch?'})
	              )
		## Reformat ratings matrix
		ratings_full = self.ratings.unstack().reset_index(name='rating')

		## get matrix for a user
		user_full = (ratings_full[ratings_full.userId == (user)].
         			sort_values('rating', ascending=False))

		## merge matrix user and prediction
		pred = user_full.merge(pred_matrix, how = 'left', on = ['userId','movieId'])

		## find movies not seen by user and find the top movies by predicted rating
		recommendations = pred[pred.rating == 0].sort_values('Must watch?', ascending=False)[:top]
		recommendations['Must watch?'] = (recommendations.apply(lambda x: self.estimate(x.userId, x.movieId), axis=1).
											astype(int).
											apply(lambda x: ''.join(u'\u2713' for _ in range(x)) if x*100 > 0 else u'\u2A09')) ##looks cooler with symbols

		## get only certain columns
		cols = ['movieId','title','original_language','genres','release_date']
		movies_data = self.movies_data[cols]

		## return recommendation with movies metadata df 
		return recommendations.merge(movies_data, on='movieId', how='inner').drop(['level_0_x','level_0_y'], axis=1, errors='ignore')


def grid_search(clf, params):
	clfs = []
	for k in params['Ks']:
	    for alpha in params['alphas']:
	        for epoch in params['epochs']:
	            clf.fit(k,epoch, alpha)
	            print(f"K: {k}\t| alpha: {alpha}\t| epochs: {epoch}\t| NDCG: {clf.score}")
	            clfs.append([k,alpha,epoch,clf.score,clf])
	return pd.DataFrame(clfs, columns=["K","alpha","epochs","NDCG","clf"])