from functools import (reduce)
import math
import numpy

def sigmoid(val):
    return 1 / ( 1 + ( math.e ** -val))

vsigmoid = numpy.vectorize(sigmoid, otypes=[numpy.float])

def sigmoid_derivative(val):
    return val * (1 - val)

vsigmoid_derivative = numpy.vectorize(sigmoid_derivative, otypes=[numpy.float])

def compute_layer(i, layer, activation=vsigmoid):
    """ activation should be a vectorized function """

    net = i.dot(layer['w']) + layer['b']
    activated = activation(net)

    #print('compute layer:', 'i:', i, 'layer:', layer, 'net:', net, 'activated:', activated)

    return activated

def feed_forward(i, net, activation=vsigmoid):
    """ activation should be a vectorized function """

    def reduce_layer(memo, layer):
        input = i if len(memo) == 0 else memo[-1]

        memo.append(compute_layer(input, layer, activation))

        return memo

    return reduce(reduce_layer, net, [])
