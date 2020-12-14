import numpy as np
import scipy
from keras.datasets import mnist
from matplotlib import pyplot as plt
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


gradients = np.zeros([len(output)].append(list(weight.shape)))

# Function to multiply everything and determine node values
def forward(image,weight):
    hiddenLayers[0] = sigmoid(np.matmul(image, weight[0]))
    hiddenLayers[1] = sigmoid(np.matmul(hiddenLayers[0], weight[1]))
    output = sigmoid(np.matmul(hiddenLayers[1], weight[2]))

    im = image.reshape(28, 28)*255
    plt.imshow(im, cmap='gray')

    return output

# Function to keep node values between 0 and 1
def sigmoid(vector):
    return 1/(1 + np.exp(-vector))


################### Maybe all of this is wrong #####################################


def sig_derivative(vector):
    return np.exp(-vector)/(1 + np.exp(-vector))**2

def partial_derivative(weight_n,j,k,gradients,weight,image,node):
    if weight_n == len(weight) - 1:
        gradients[node][weight_n][j][k] = hiddenLayers[1][j] * \
                                          sig_derivative(np.dot(hiddenLayers[1], weight[weight_n][j]))

    else:
        if weight_n > 0:
            g = 0
            for l in range(0, len(hiddenLayers[weight_n])):
                g = g + hiddenLayers[weight_n][l] * \
                    sig_derivative(np.dot(hiddenLayers[weight_n], weight[weight_n + 1][l])) * \
                    partial_derivative(weight_n + 1, j,k,gradients,weight,image,node)
            gradients[node][weight_n][j][k] = g

        if weight_n == 0:
            g = 0
            for l in range(0, len(image[weight_n])):
                g = g + image[weight_n - 1][l] * \
                    sig_derivative(np.dot(image[l], weight[weight_n + 1][l])) * \
                    partial_derivative(weight_n + 1, j, k, gradients, weight,image,node)
            gradients[node][weight_n][j][k] = g
    return gradients[node][weight_n][j][k]


def back_propagate(image,weight,output):
    #gradients = np.zeros((num_out,weight[0].shape[0] * weight[0].shape[1] + weight[1].shape[0] * weight[1].shape[1] +
     #                    weight[2].shape[0] * weight[2].shape[1]))

    for node in range(0,len(output)):
        for weight_n in range(0,len(weight)):
            for j in range(0,len(weight[weight_n])):
                for k in range(0,len(weight[weight_n][j])):
                    gradients[node][weight_n][j][k] = partial_derivative(weight_n, j, k, gradients, weight,image,node)


                    ############ Testing ###################
axes=[]
fig=plt.figure()

for i in range(0, 5):
    axes.append(fig.add_subplot(1, 5, i + 1))
    print(forward(x_train[i], weights))

fig.tight_layout()
plt.show()
