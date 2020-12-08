import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=RuntimeWarning) 

class matrix_factorization():

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

		summ = 0
		for col in range(self.movies):
			summ += self.gradient(user_row,col,user)
		return summ/self.movies


	def V_gradient(self,movie_col,movie):

		summ = 0
		for row in range(self.users):
			summ += self.gradient(row,movie_col,movie)
		return summ/self.users


	def update_U(self):
						
		for i in range(self.users):
			for j in range(self.K):
				self.U[i,j] += self.learning_rate * self.U_gradient(i,j)


	def update_V(self):
		
		for i in range(self.K):
			for j in range(self.movies):
				self.V[i,j] += self.learning_rate * self.V_gradient(j,i)


	def fit(self,learning_rate=0.1, iterations=2):
		
		self.learning_rate = learning_rate
		self.mse = []
		for i in range(iterations):
			self.update_U()
			self.update_V()
			self.mse.append(self.MSE)


	def top_recommends(self, user, top=5, movies_data):
		predictions = pd.DataFrame(
									self.matrix, 
								   	columns = self.ratings.columns, 
								   	index = self.ratings.index).round()
		predictions = predictions.unstack().reset_index(name='rating')
		predictions = predictions.set_index('userId').sort_index(axis = 0)
		try:
			recommends = predictions.loc[user].groupby('movieId').first().rating.nlargest(top).reset_index(name='rating')
			return recommends.merge(movies_data, on='movieId', how='inner')
		except Exception as e:
			return (f"User: {e} not found")


	@property
	def MSE(self):
				
		matrix_product = np.matmul(self.U, self.V)
		return np.sum((self.data - matrix_product)**2)

	@property
	def matrix(self):
		return np.dot(self.U, self.V)


def find_best(data, max_Ks=10, learning_rate=[0.001,0.01,0.1,1], iterations=2):
	mse = []
	for k in range(1, max_Ks+1):
	    for lr in (learning_rate):
	        r = matrix_factorization(data,k)
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