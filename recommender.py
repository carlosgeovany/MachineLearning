import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass

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
	:K:	int
		hyperparameter to iterate over
	:lambd:	int
		hyperparameter
	:users:	int
		total number of users
	:movies:	int
		total number of movies
	:U: numpy array
		random array with shape of users an K´s
	:V:	numpy array
		random array with shape of K´s and movies

	Methods
	-------

	gradient(user_row, movie_col, user=None, movie=None)
		calculates gradient descent between Uij and Vji

	U_gradient(self,user_row,user)
		calls gradient with user rows and accumulates the results

	"""

	def __init__(self,data,K):
		self.ratings = data
		self.data = self.ratings.to_numpy()
		self.K = K
		self.lambd = 1
		self.users = data.shape[0]
		self.movies = data.shape[1]
		self.U = np.random.uniform(low=0.1, high=0.9, size=(self.users, self.K))
		self.V = np.random.uniform(low=0.1, high=0.9, size=(self.K, self.movies))


	def gradient(self, user_row, movie_col, user=None, movie=None):
		"""
		calculates gradient descent between Uij and Vji

		:user_row: one user row with movies ratings
		:movie_col:one movie_col with users ratings
		:user: to match condition for calculate gradient on users
		:movie: to match condition for calculate gradient on movies

		:return: calculated gradient
		"""
		rating = float(self.data[user_row,movie_col])
		prediction = float(np.dot(self.U[user_row,:],self.V[:,movie_col]))

		if user != None:
			rows = float(self.V[:,movie_col][user])
			grad = 0.5*((rating - prediction) + self.lambd/2) * rows
		else:
			columns = float(self.U[user_row,:][movie])
			grad = 0.5*((rating - prediction) + self.lambd/2) * columns
		return grad


	def U_gradient(self,user_row,user):
		"""
		calls gradient with user rows and accumulates the results

		:user_row:	array
			array with all users
		:user:	int
			one user

		:return: summ of gradients
		"""
		summ = 0
		for col in range(self.movies):
			summ += self.gradient(user_row,col,user)
		return summ/self.movies


	def V_gradient(self,movie_col,movie):
		"""
		calls gradient with movie column and accumulates the results

		:movie_col:	array
		:movie: int
			one movie

		:return: summ of gradients
		"""
		summ = 0
		for row in range(self.users):
			summ += self.gradient(row,movie_col,movie)
		return summ/self.users


	def update_U(self):
		"""
		update users matrix
		"""	
		for i in range(self.users):
			for j in range(self.K):
				self.U[i,j] += self.learning_rate * self.U_gradient(i,j)


	def update_V(self):
		"""
		update moves matrix
		"""
		for i in range(self.K):
			for j in range(self.movies):
				self.V[i,j] += self.learning_rate * self.V_gradient(j,i)


	def fit(self,learning_rate=0.1, iterations=2):
		"""
		train the recommender

		:learning_rate: float
			hyperparameter
		:iterations: int
			how many iterations should we do to train
		"""
		self.learning_rate = learning_rate
		self.mse = []
		for i in range(iterations):
			self.update_U()
			self.update_V()
			self.mse.append(self.MSE)


	def top_recommends(self, user, movies_data, top=5):
		"""
		movies recommended for a user

		:user:	int
			a single user to recommend
		:movies_data:	pandas df
			df with movies metadata
		:top:	int
			max number of movies to be recommended
		:return:	pandas df
			df with movies recommended for a user
		"""
		## Reformat full matrix
		pred_matrix = (pd.DataFrame(self.matrix, columns = self.ratings.columns, index = self.ratings.index).
	               round().
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
		recommendations['Must watch?'] = (recommendations['Must watch?'].
					astype(int).
					apply(lambda x: ''.join(u'\u2713' for _ in range(x)) if x > 0 else u'\u2A09')) ##looks cooler with symbols

		## get only certain columns
		cols = ['movieId','title','original_language','genres','release_date']
		movies_data = movies_data[cols]

		## return recommendation with movies metadata df 
		return recommendations.merge(movies_data, on='movieId', how='inner')


	@property
	def MSE(self):
		"""
		:return: float
			MSE for matrix product U*V 
		"""
				
		matrix_product = np.matmul(self.U, self.V)
		return np.sum((self.data - matrix_product)**2)

	@property
	def matrix(self):
		"""
		:return: float
			dot product U*V
		"""
		return np.dot(self.U, self.V)


def find_best(data, max_Ks=10, learning_rate=[0.001,0.01,0.1,1], iterations=2):
	"""
	method to find best hyperparameters
	:return: pandas df
		df with hyperparameters calculated
	"""
	mse = []
	for k in range(1, max_Ks+1):
	    for lr in (learning_rate):
	        r = MatrixFactorization(data,k)
	        r.fit(lr, iterations)
	        me = np.mean(r.mse)
	        mse.append([k,lr,me])
	        if max_Ks < 10:
	        	print(f"K: {k}\t | Learning_rate: {lr}\t | MSE:{me}")
	        else:
	        	if k % 10 == 0:
	        		print(f"K: {k}\t | Learning_rate: {lr}\t | MSE:{me}")
	mse = pd.DataFrame(mse, columns=["K","Learning_rate","MSE"])
	return mse