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

    return activation(net)

def feed_forward(i, net, activation=vsigmoid):
    """ activation should be a vectorized function """

    def reduce_layer(memo, layer):
        input = i if len(memo) == 0 else memo[-1]

        memo.append(compute_layer(input, layer, activation))

        return memo

    return reduce(reduce_layer, net, [])

def init_net(dataset, layer_sizes):
    complete_layer_sizes = ([dataset['num_inputs']] +
        layer_sizes +
        [dataset['num_outputs']])

    def init_layer(upper_index):
        weight_shape = (complete_layer_sizes[upper_index-1],
                complete_layer_sizes[upper_index])
        bias_shape = (1, complete_layer_sizes[upper_index])

        return {'w': (numpy.random.random(weight_shape) - .5) / 2,
                'b': (numpy.random.random(bias_shape) -.5) / 2 }

    return [ init_layer(i) for i in range(1, len(complete_layer_sizes)) ]

     

