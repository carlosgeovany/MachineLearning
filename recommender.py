import numpy as np
from dataclasses import dataclass


class matrix_factorization():

	def __init__(self,data,K):
		self.data = data
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


	def user_gradient(self,user_row,user):

		summ = 0
		for col in range(self.movies):
			summ += self.gradient(user_row,col,user)
		return summ/self.movies


	def movie_gradient(self,movie_col,movie):

		summ = 0
		for row in range(self.users):
			summ += self.gradient(row,movie_col,movie)
		return summ/self.users


	def update_U(self):
						
		for i in range(self.users):
			for j in range(self.K):
				self.U[i,j] += self.learning_rate * self.user_gradient(i,j)


	def update_V(self):
		
		for i in range(self.K):
			for j in range(self.movies):
				self.V[i,j] += self.learning_rate * self.movie_gradient(j,i)


	def fit(self,learning_rate=0.1, iterations=2):
		
		self.learning_rate = learning_rate
		self.mse = []
		for i in range(iterations):
			self.update_U()
			self.update_V()
			self.mse.append(self.MSE)


	@property
	def MSE(self):
				
		matrix_product = np.matmul(self.U, self.V)
		return np.sum((self.data - matrix_product)**2)

	@property
	def matrix(self):
		return np.dot(self.U, self.V)

