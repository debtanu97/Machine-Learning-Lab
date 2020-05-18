import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt 
import inspect

xpt=[]
ypt=[]

''' 
You are only required to fill the following functions
mean_squared_loss
mean_squared_gradient
mean_absolute_loss
mean_absolute_gradient
mean_log_cosh_loss
mean_log_cosh_gradient
root_mean_squared_loss
root_mean_squared_gradient
preprocess_dataset
main

Don't modify any other functions or commandline arguments because autograder will be used
Don't modify function declaration (arguments)

'''

def mean_squared_loss(xdata, ydata, weights):
	'''
	weights = weight vector [D X 1]
	xdata = input feature matrix [N X D]
	ydata = output values [N X 1]
	Return the mean squared loss
	'''

	loss = ydata - np.matmul(xdata,weights)
	#print loss
	sq_loss = np.dot(loss,loss)
	#print sq_loss
	sq_loss = sq_loss/(len(ydata))	

	return sq_loss

	#raise NotImplementedError

def mean_squared_gradient(xdata, ydata, weights):
	'''
	weights = weight vector [D X 1]
	xdata = input feature matrix [N X D]
	ydata = output values [N X 1]
	Return the mean squared gradient
	'''

	loss = np.matmul(xdata,weights) - ydata
	loss = np.transpose(loss)
	loss = 2*loss
	grad = np.matmul(loss, xdata)
	grad = grad/(len(ydata))

	return grad

	#raise NotImplementedError

def mean_absolute_loss(xdata, ydata, weights):
	loss = ydata - np.matmul(xdata,weights)
	loss = abs(loss)/(len(ydata))	
	loss = np.sum(loss)

	return loss	
	#raise NotImplementedError

def mean_absolute_gradient(xdata, ydata, weights):
	loss = ydata - np.matmul(xdata,weights)
	absloss = abs(loss)
	loss = (np.divide(loss, absloss, out=np.zeros_like(loss), where=absloss!=0))
	loss = np.transpose(loss)
	xterm = (-1)*xdata
	grad = np.matmul(loss, xterm)

	return grad/len(ydata)    
	#raise NotImplementedError

def mean_log_cosh_loss(xdata, ydata, weights):
	loss = ydata - np.matmul(xdata,weights)
	temp = [80.0]*len(ydata)
	temp = np.array(temp)
	loss = np.minimum(temp,abs(loss))
	#print(loss)
	loss = np.cosh(loss)
	loss = np.log(loss)
	loss = np.sum(loss)

	return loss/len(ydata)
	#raise NotImplementedError

def mean_log_cosh_gradient(xdata, ydata, weights):
	loss = ydata - np.matmul(xdata,weights)
	loss = np.tanh(loss)
	loss = np.transpose(loss)
	xterm = (-1)*xdata
	grad = np.matmul(loss, xterm)
	grad = grad/len(ydata)

	return grad

	#raise NotImplementedError

def root_mean_squared_loss(xdata, ydata, weights):

	loss = ydata - np.matmul(xdata,weights)
	#print loss
	sq_loss = np.dot(loss,loss)
	#print sq_loss
	m_sq_loss = sq_loss/(len(ydata))	
	
	rms_loss = np.sqrt(m_sq_loss)

	return rms_loss

	#raise NotImplementedError

def root_mean_squared_gradient(xdata, ydata, weights):
	rms_loss = root_mean_squared_loss(xdata, ydata, weights)

	loss = np.matmul(xdata,weights) - ydata
	loss = np.transpose(loss)
	grad = np.matmul(loss, xdata)
	grad = grad/len(ydata)

	rms_grad = np.true_divide(grad, rms_loss)

	return  rms_grad

	#raise NotImplementedError

class LinearRegressor:

	def __init__(self,dims):
		
		# dims is the number of the features
		# You can use __init__ to initialise your weight and biases
		# Create all class related variables here
		np.random.seed(4)
		self.dims = dims	
		self.weights = np.random.rand(dims)	
		#raise NotImplementedError

	def train(self, xtrain, ytrain, loss_function, gradient_function, epoch=100, lr=1.0):
		'''
		xtrain = input feature matrix [N X D]
		ytrain = output values [N X 1]
		learn weight vector [D X 1]
		epoch = scalar parameter epoch
		lr = scalar parameter learning rate
		loss_function = loss function name for linear regression training
		gradient_function = gradient name of loss function
		'''
		# You need to write the training loop to update weights here
		#print self.weights
		
		#Graph Plotting
		global xpt
		global ypt
		xpt=[]
		ypt=[]

		for i in range(epoch) :
			grad = np.zeros(self.dims, dtype='float')
			error = -1
			
			error = loss_function(xtrain, ytrain, self.weights)
			grad = gradient_function(xtrain, ytrain, self.weights)
			#print grad
			
			self.weights = self.weights - lr*grad
			#print self.weights 
			
			#error = loss_function(xtrain, ytrain, self.weights)
			#print (error)

			#Plotting
			xpt.append(i)
			ypt.append(error)

		#raise NotImplementedError

	def predict(self, xtest):
		
		# This returns your prediction on xtest
		wt = np.transpose(self.weights)
		y_pred = np.matmul(xtest, wt)
		return np.array(y_pred.astype(int))

		#raise NotImplementedError	


def read_dataset(trainfile, testfile):
	'''
	Reads the input data from train and test files and 
	Returns the matrices Xtrain : [N X D] and Ytrain : [N X 1] and Xtest : [M X D] 
	where D is number of features and N is the number of train rows and M is the number of test rows
	'''
	xtrain = []
	ytrain = []
	xtest = []

	with open(trainfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtrain.append(row[:-1])
			ytrain.append(row[-1])

	with open(testfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtest.append(row)

	return np.array(xtrain), np.array(ytrain), np.array(xtest)

def preprocess_dataset(xdata, ydata=None):
	
	#xdata = input feature matrix [N X D] 
	#ydata = output values [N X 1]
	#Convert data xdata, ydata obtained from read_dataset() to a usable format by loss function

	#The ydata argument is optional so this function must work for the both the calls
	#xtrain_processed, ytrain_processed = preprocess_dataset(xtrain,ytrain)
	#xtest_processed = preprocess_dataset(xtest)	
	
	

	x_0 = np.ones(xdata.shape[0], dtype='float')
	x_0 = np.array(x_0)[np.newaxis]
	x_0 = np.transpose(x_0)

	#remove Date and id no.
	xdata = np.delete(xdata,[0,1],1) 

	#add leading 1's to compensate for constant term
	xdata = np.hstack((x_0,xdata))
		
	#delete the weekday attr.
	xdata = np.delete(xdata,4,1)

	xdata = xdata.astype(float)

	if(ydata is not None):
		ydata = ydata.astype(float)

	#hot encoding of season
	season = xdata[ : , 1]
	s1=[]
	s2=[]
	s3=[]
	s4=[]
	for i in range(0,len(xdata)):
		if(season[i] == 1) :
			s1=np.append(s1,1) 
			s2=np.append(s2,0) 
			s3=np.append(s3,0) 
			s4=np.append(s4,0) 

		elif(season[i] == 2) :
			s1=np.append(s1,0) 
			s2=np.append(s2,1) 
			s3=np.append(s3,0) 
			s4=np.append(s4,0) 

		elif(season[i] == 3) :
			s1=np.append(s1,0) 
			s2=np.append(s2,0) 
			s3=np.append(s3,1) 
			s4=np.append(s4,0) 

		elif(season[i] == 4):
			s1=np.append(s1,0) 
			s2=np.append(s2,0) 
			s3=np.append(s3,0) 
			s4=np.append(s4,1) 


	xdata = np.delete(xdata,1,1)
	
	s1 = np.array(s1)[np.newaxis]
	s2 = np.array(s2)[np.newaxis]
	s3 = np.array(s3)[np.newaxis]
	s4 = np.array(s4)[np.newaxis]
	
	s1 = np.transpose(s1)
	s2 = np.transpose(s2)
	s3 = np.transpose(s3)
	s4 = np.transpose(s4)

	xdata = np.hstack((xdata,s1))
	xdata = np.hstack((xdata,s2))
	xdata = np.hstack((xdata,s3))
	xdata = np.hstack((xdata,s4))

	#print(xdata[0])
	#Hot encoding of hour
	temp=[]
	v=[0.0]*24
	hour = xdata[:,1]

	for i in hour :
		v[int(i)] = 1.0
		b=v.copy()
		temp.append(b)
		v[int(i)] = 0.0


	temp = np.array(temp, dtype='float64')		
	xdata = np.hstack((xdata,temp))
	xdata = np.delete(xdata, 1, axis=1)
	
	#hot encoding of weather
	season = xdata[ : , 3]
	#print season
	s1=[]
	s2=[]
	s3=[]
	s4=[]
	for i in range(0,len(xdata)):
		if(season[i] == 1.0) :
			s1=np.append(s1,1) 
			s2=np.append(s2,0) 
			s3=np.append(s3,0) 
			s4=np.append(s4,0) 

		elif(season[i] == 2.0) :
			s1=np.append(s1,0) 
			s2=np.append(s2,1) 
			s3=np.append(s3,0) 
			s4=np.append(s4,0) 

		elif(season[i] == 3.0) :
			s1=np.append(s1,0) 
			s2=np.append(s2,0) 
			s3=np.append(s3,1) 
			s4=np.append(s4,0) 

		elif(season[i] == 4.0):
			s1=np.append(s1,0) 
			s2=np.append(s2,0) 
			s3=np.append(s3,0) 
			s4=np.append(s4,1) 


	xdata = np.delete(xdata,3,1)
	
	s1 = np.array(s1)[np.newaxis]
	s2 = np.array(s2)[np.newaxis]
	s3 = np.array(s3)[np.newaxis]
	s4 = np.array(s4)[np.newaxis]
	
	s1 = np.transpose(s1)
	s2 = np.transpose(s2)
	s3 = np.transpose(s3)
	s4 = np.transpose(s4)

	xdata = np.hstack((xdata,s1))
	xdata = np.hstack((xdata,s2))
	xdata = np.hstack((xdata,s3))
	xdata = np.hstack((xdata,s4))



	if(ydata is not None):
		return np.array(xdata), np.array(ydata)

	else :	
		return np.array(xdata)	
	

	#NOTE: You can ignore/drop few columns. You can feature scale the input data before processing further.
	#raise NotImplementedError

dictionary_of_losses = {
	'mse':(mean_squared_loss, mean_squared_gradient),
	'mae':(mean_absolute_loss, mean_absolute_gradient),
	'rmse':(root_mean_squared_loss, root_mean_squared_gradient),
	'logcosh':(mean_log_cosh_loss, mean_log_cosh_gradient),
}

def main():

	# You are free to modify the main function as per your requirements.
	# Uncomment the below lines and pass the appropriate value

	xtrain, ytrain, xtest = read_dataset(args.train_file, args.test_file)
	xtrainprocessed, ytrainprocessed = preprocess_dataset(xtrain, ytrain)
	xtestprocessed = preprocess_dataset(xtest)
	

	model = LinearRegressor(xtrainprocessed.shape[1])
	
	#x = np.zeros(xtrainprocessed.shape[1], dtype='float')
	#print(mean_absolute_loss(xtrainprocessed, ytrainprocessed, x))


	# The loss function is provided by command line argument	
	loss_fn, loss_grad = dictionary_of_losses[args.loss]
	model.train(xtrainprocessed, ytrainprocessed, loss_fn, loss_grad, args.epoch, args.lr)
	
	# Graph Plotting
	#plt.xlabel('epoch')
	#plt.ylabel('mean_squared_error')
	
	#model.train(xtrainprocessed, ytrainprocessed, mean_squared_loss, mean_squared_gradient, args.epoch, args.lr)
		
	#plt.plot(xpt,ypt,'-r',label='mean_squared_gradient')
	#plt.legend()
	
	#model = LinearRegressor(xtrainprocessed.shape[1])
	#model.train(xtrainprocessed, ytrainprocessed, mean_squared_loss, mean_absolute_gradient, args.epoch, args.lr)
	#plt.plot(xpt,ypt,'-b',label='mean_absolute_gradient')	
	#plt.legend()
	
	#model = LinearRegressor(xtrainprocessed.shape[1])
	#model.train(xtrainprocessed, ytrainprocessed, mean_squared_loss, root_mean_squared_gradient, args.epoch, args.lr)
	#plt.plot(xpt,ypt,label='root_mean_squared_gradient')	
	#plt.legend()

	#model = LinearRegressor(xtrainprocessed.shape[1])
	#model.train(xtrainprocessed, ytrainprocessed, mean_squared_loss, mean_log_cosh_gradient, args.epoch, args.lr)
	#plt.plot(xpt,ypt,label='mean_log_cosh_gradient')	
	#plt.legend()
	#plt.show()

	# Prediction
	ytest = model.predict(xtestprocessed)

	ytest_p=[["instance (id)" , "count"]]

	for i in range(len(ytest)):
		if(ytest[i] < 0):
			ytest_p.append([i,0])
		else:
			ytest_p.append([i,ytest[i]])	


	for i in ytest_p:
		print(str(i[0])+","+str(i[1]))

	#Writing in csv file
	#with open('prediction.csv', 'w') as csvFile:
	#	writer = csv.writer(csvFile)
	#	writer.writerows(ytest_p)

	#csvFile.close()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--loss', default='mse', choices=['mse','mae','rmse','logcosh'], help='loss function')
	parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
	parser.add_argument('--epoch', default=100000, type=int, help='number of epochs')
	parser.add_argument('--train_file', type=str, help='location of the training file')
	parser.add_argument('--test_file', type=str, help='location of the test file')

	args = parser.parse_args()

	main()
