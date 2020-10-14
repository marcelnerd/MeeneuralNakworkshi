import numpy as np
import scipy
from keras.datasets import mnist

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

