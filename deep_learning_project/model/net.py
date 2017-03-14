import math
import numpy

def sigmoid(val):
    return 1 / ( 1 + ( math.e ** -val))

vsigmoid = numpy.vectorize(sigmoid, otypes=[numpy.float])

def sigmoid_derivative(val):
    return val * (1 - val)

vsigmoid_derivative = numpy.vectorize(sigmoid_derivative, otypes=[numpy.float])

def compute_layer(i, w, b, activation=vsigmoid):
    """ activation should be a vectorized function """

    net = i.dot(w) + b

    return activation(net)

def feed_forward(i, w1, b1, w2, b2, activation=vsigmoid):
    """ activation should be a vectorized function """

    z1 = compute_layer(i, w1, b1, activation)
    z2 = compute_layer(z1, w2, b2, activation)

    return (z1, z2)
