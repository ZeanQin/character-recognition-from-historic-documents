# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
from numpy import genfromtxt
import csv
import pandas as pd


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# load the raw training data from a csv file into a numpy list
print("[INFO loading the training data ...")
my_data = genfromtxt('train.csv', delimiter=',')

# reshape the training dataset from a flat list of 429-dim vectors, to
# 13 x 33 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
data = my_data[:, 9:] 


data = data.reshape((data.shape[0], 33, 13))
target = my_data[:, 1]

print("[INFO Data Augmentation ...")

target = np.concatenate((target, target, target, 
	target, target, target, target))
emp_row = np.zeros(shape = (1, 13))
emp_col = np.zeros(shape = (33, 1))

pool1 = np.zeros(shape = data.shape)

for i in range(0, data.shape[0]):
    img = data[i,:,:]
    new_img = np.vstack((emp_row, img))
    x = np.delete(new_img, (33), axis=0)
    x = x.reshape(1, 33, 13)
    pool1[i,:,:] = x
print("[INFO 1st Artifical Dataset Generated ...")   

                 
pool2 = np.zeros(shape=data.shape)
for i in range(0, data.shape[0]):
    img = data[i,:,:]
    new_img = np.vstack((emp_row, emp_row, img))
    x = np.delete(new_img, (33,34), axis=0)
    x = x.reshape(1, 33, 13)
    pool2[i,:,:] = x
print("[INFO 2nd Artifical Dataset Generated ...")


pool3 = np.zeros(shape=data.shape)
for i in range(0, data.shape[0]):
    img = data[i,:,:]
    new_img = np.vstack((img, emp_row))
    x = np.delete(new_img, (0), axis=0)
    x = x.reshape(1, 33, 13)
    pool3[i,:,:] = x
print("[INFO 3rd Artifical Dataset Generated ...")


pool4 = np.zeros(shape=data.shape)
for i in range(0, data.shape[0]):
    img = data[i,:,:]
    new_img = np.vstack((img, emp_row, emp_row))
    x = np.delete(new_img, (0,1), axis=0)
    x = x.reshape(1, 33, 13)
    pool4[i,:,:] = x
print("[INFO 4th Artifical Dataset Generated ...")


pool5 = np.zeros(shape = data.shape)
for i in range(0, data.shape[0]):
    img = data[i,:,:]
    new_img = np.hstack((img, emp_col))
    x = np.delete(new_img, (0), axis=1)
    x = x.reshape(1, 33, 13)
    pool5[i,:,:] = x
print("[INFO 5th Artifical Dataset Generated ...")

pool6 = np.zeros(shape = data.shape)
for i in range(0, data.shape[0]):
    img = data[i,:,:]
    new_img = np.hstack((img, emp_col, emp_col))
    x = np.delete(new_img, (0,1), axis=1)
    x = x.reshape(1, 33, 13)
    pool6[i,:,:] = x
print("[INFO 6th Artifical Dataset Generated ...")




data = np.concatenate((data, pool1, pool2, pool3, 
	pool4, pool5, pool6))
print("[INFO Data Augmentation Completed ...")


print("[INFO the shape of data is ...")
print(data.shape)
# print data
data = data[:, np.newaxis, :, :]
# print data
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data / 255.0, target.astype("int"), test_size=0.1)
# print "trainData", trainData
# print "testData", testData
# print "trainLabels", trainLabels
# print "testLabels", testLabels

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 98)
testLabels = np_utils.to_categorical(testLabels, 98)


# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=13, height=33, depth=1, classes=98,
	weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(trainData, trainLabels, batch_size=128, nb_epoch=20,
		verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

# load the test data from csv
print("[INFO] saving result to file...")
my_test_data = genfromtxt('test.csv', delimiter=',')
test_data = my_test_data[:, 9:]


test_data = test_data.reshape((test_data.shape[0], 33, 13))
test_data = test_data[:, np.newaxis, :, :]
test_data = test_data / 255.0
result = model.predict(test_data)
result = result.argmax(axis=1)
print result


print("[INFO] done ...")

myfile = open('result.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(result)

