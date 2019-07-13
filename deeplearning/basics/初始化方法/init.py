import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


def initialize_parameters_random(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def initialize_parameters_sqrt(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        mu = 0
        sigma = np.sqrt(1.0 / layers_dims[l - 1])
        parameters['W' + str(l)] = np.random.normal(loc=mu, scale=sigma, size=(layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def initialize_parameters_xavier(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        mu = 0
        sigma = np.sqrt(2.0 / (layers_dims[l - 1] + layers_dims[l]))
        parameters['W' + str(l)] = np.random.normal(loc=mu, scale=sigma, size=(layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def initialize_parameters_kaiming(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        mu = 0
        sigma = np.sqrt(2.0 / layers_dims[l - 1])
        parameters['W' + str(l)] = np.random.normal(loc=mu, scale=sigma, size=(layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


def relu(Z):
    A = np.maximum(0, Z)
    return A
def sigmoid(Z):
    A = 1.0 / (1 + np.exp(-Z))
    return A
def tanh(Z):
    A = np.tanh(Z)
    return A

def forward_propagation(initialization='kaiming'):
    data = np.random.randn(1000, 10000)
    # layers_dims = [1000, 800, 500, 300, 200, 100, 90, 80, 40, 20, 10]
    layers_dims = [1000] * 11
    num_layers = len(layers_dims)
    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'sqrt':
        parameters = initialize_parameters_sqrt(layers_dims)
    elif initialization == 'xavier':
        parameters = initialize_parameters_xavier(layers_dims)
    elif initialization == 'kaiming':
        parameters = initialize_parameters_kaiming(layers_dims)

    A = data
    for l in range(1, num_layers):
        A_pre = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        z = np.dot(W, A_pre) + b # z = Wx + b

        # 不同的激活函数适用不同的初始化方法
        # A = sigmoid(z)
        # A = tanh(z)
        A = relu(z)

        print(A)
        plt.subplot(2, 5, l)
        plt.hist(A.flatten(), facecolor='g')
        # plt.xlim([-1, 1])
        plt.xlim([-5, 5])
        plt.ylim([0, 1000000])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    '''
    实验不同初始化方法在不同激活函数下的表现
    '''
    # forward_propagation('zeros')
    # forward_propagation('random')
    # forward_propagation('sqrt')
    # forward_propagation('xavier')
    forward_propagation('kaiming')
