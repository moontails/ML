import numpy
import os.path
import sys
import random

#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus 
def readFile(f):
	# Reads in the modified text file in ARFF format which contains one instance per line.	
	if os.path.isfile(f):
		file = open(f, "r") # open the input file in read-only mode
		#Features - As a list of lists
		x = []
		#Label - Negative as -1, Positive as +1
		y = []
		print("\nReading file ", f)
		#Parse file to get Features Values and Label Value
		for line in file:
			temp = line.rstrip('\n').split(',')
			x.append([float(i) for i in temp[:-1]])
			y.append(float(temp[-1]))
		return x,y
	else:
		print("\nError: corpus file ", f, " does not exist")  # For simplicity's sake, this will suffice.
		sys.exit() # exit the script

#----------------------------------------
# The SGD Function
#----------------------------------------
def SGD(x,y,w,alpha,iterations):
	od = 0.0
	temp = 0.0

	#GD - run for iter=5000 and alpha=.001
	for i in range(0,iterations):
		print "\nRunning for epoch: " + str(i)
		for j in range(0,len(x)):
			od = numpy.dot(w,x[j])
			temp = alpha*(y[j]-od)
			templ=list(numpy.multiply(temp,x[j]))
			w = list(numpy.add(w,templ))
	return w

#----------------------------------------
# Function to do disjoint sampling from the large instance space
#----------------------------------------
def sampler(x,y,n):
	temp1 = x
	temp2 = y
	X = []
	for k in range(0,len(x)/n):
		x_sample = []
		y_sample = []
		for i in range(0,n):
			sample=random.choice(temp1)
			j=temp1.index(sample)
			y_sample.append(temp2[j])
			x_sample.append(sample)
			temp1.pop(j)
			temp2.pop(j)
		X.append((x_sample,y_sample))
	return X

#----------------------------------------
# Function to compute accuracy
#----------------------------------------
def accuracy(x,y,w):
	pred = []
	for i in range(0,len(x)):
		c = numpy.dot(w,x[i])
		if (c >= 0):
			pred.append(1.0)
		else:
			pred.append(-1.0)
	count = 0
	for i in range(0,len(pred)):
		if (pred[i] == y[i]):
			count+=1
	pa = float(count)/len(pred)
	return pa

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
	#Features, Label
	x,y = readFile('train')
	#Weight
	w = []
	#setting learning rate
	alpha = 0.001

	#setting convergence threshold
	epsilon=0.001

	#Number of iterations to run
	iterations = 500
	
	#Accuracy List
	pa = []
	#Normalizing x 
	for i in x:
		i.append(1)

	#Initialize the weight vector to zero and normalizing w to include theta
	for i in range( 0, len(x[0]) ):
		w.append(0.0)
	
	#Sample and get the five choices of the test set
	X = sampler(x,y,400)
	
	for i in range(0,len(X)):
		w=SGD(X[i][0],X[i][1],w,alpha,iterations)
		w=SGD(X[(i+1)%5][0],X[(i+1)%5][1],w,alpha,iterations)
		w=SGD(X[(i+2)%5][0],X[(i+2)%5][1],w,alpha,iterations)
		w=SGD(X[(i+3)%5][0],X[(i+3)%5][1],w,alpha,iterations)		
		#Use the fifth sample for accuracy prediction
		pa.append(accuracy(X[(i+4)%5][0],X[(i+4)%5][1],w))
	
	#Calculating mean pred accuracy over the five folds
	mean_pa = sum(pa)/float(len(pa))

	#Final training on entire set of examples
	w=SGD(x,y,w,alpha,500)
	
	print "\nThe pa for each fold"
	print pa
	print "\nThe mean pa is : " + str(mean_pa)
	print "\nThe weight vector after learning\n"
	print w
