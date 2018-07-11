import numpy as np
from log_regex_utils import load_dataset

def process_data():
	train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
	#print(train_set_x_orig.shape)

	train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
	test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T
	print(train_x_flatten.shape)
	print(test_x_flatten.shape)

	#staandardize images	
	train_x = train_x_flatten/255
	test_x  = test_x_flatten/255

	return train_x, train_y, test_x, test_y 

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def init_params(dim):
	w = np.zeros((dim,1))
	b = 0
	return w,b

def forward_prop(w, b, X, Y):
	m = X.shape[1]

	z = np.dot(w.T, X) + b
	A = sigmoid(z)

	cost = -1/m * np.sum( Y*np.log(A) + (1-Y)*np.log(1-A) ) 
	return A, cost

def backprop(A, X, Y):
	m = X.shape[1]
	dz = A-Y
	dw = 1/m * np.dot(X, dz.T)
	db = 1/m * np.sum(dz)
	return dw, db

def optimize(w, b, X, Y, iters, learning_rate):
	costs = []

	for i in range(iters):
		A,cost = forward_prop(w, b, X, Y)
		dw,db = backprop(A, X, Y)

		#update params
		w = w - (learning_rate*dw)
		b = b - (learning_rate*db)

		if(i%100==0):
			costs.append(cost)
			print("cost after iteration %i : %f "%(i,cost))

	return w, b, costs

def predict(w, b, X):
	m = X.shape[1]

	Y_prediction = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)
	Z = np.dot(w.T,X) + b
	A = sigmoid(Z)
	for i in range(A.shape[1]):
		if(A[0,i]>0.5):
			Y_prediction[0,i] = 1
		else:
			Y_prediction[0,i] = 0
	return Y_prediction	  				
		
def model(X_train, Y_train, X_test, Y_test, num_iters = 2000, learning_rate = 0.5):

	#init params
	w,b = init_params(X_train.shape[0])

	w, b, costs = optimize(w,b, X_train, Y_train, num_iters, learning_rate)

	Y_prediction_train = predict(w,b,X_train)
	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))


def main():
	
	X_train, Y_train, X_test, Y_test = process_data()
	model(X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
	main()
