import numpy as np
import scipy
from keras.datasets import mnist
import math

num_n = 16
num_out = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training data shape: ", x_train.shape) # (60000, 28, 28) -- 60000 images, each 28x28 pixels
print("Test data shape", x_test.shape) # (10000, 28, 28) -- 10000 images, each 28x28

# Flatten the images
image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

x_train = x_train/255
x_test = x_test/255

hiddenLayers = np.zeros((2,num_n))

#weights = np.random((3,len(x_train),len(hiddenLayers[0])))
weights = []
#print(weights)
weights.append(np.random.randn(image_vector_size, num_n))
weights.append(np.random.randn(num_n, num_n))
weights.append(np.random.randn(num_n, num_out))
weights = np.asarray(weights)
#print(weights.shape)
#print(weights)

# Function to multiply everything and determine node values
def forward(image,weight):
    global hiddenLayers

    hiddenLayers[0] = sigmoid(np.matmul(image,weight[0]))
    hiddenLayers[1] = sigmoid(np.matmul(hiddenLayers[0],weight[1]))
    output = sigmoid(np.matmul(hiddenLayers[1],weight[2]))
    return output

# Function to keep node values between 0 and 1
def sigmoid(vector):
    return 1/(1 + np.exp(-vector))

print(forward(x_train[0],weights))