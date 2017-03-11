import math
import numpy

# The layer array should store weights and bias

def sigmoid(val):
    return 1 / ( 1 + ( math.e ** -val))

vsigmoid = numpy.vectorize(sigmoid, otypes=[numpy.float])

def sigmoid_derivative(val):
    return val * (1 - val)

vsigmoid_derivative = numpy.vectorize(sigmoid_derivative, otypes=[numpy.float])

def compute_layer(i, w, b, activation=vsigmoid):
    """ activation should be a vectorized function """

    net = i.dot(w) + b

    return (net, activation(net))

def feed_forward(i, w1, b1, w2, b2, activation=vsigmoid):
    """ activation should be a vectorized function """

    (net1, z1) = compute_layer(i, w1, b1, activation)
    (net2, z2) = compute_layer(z1, w2, b2, activation)

    return (net1, z1, net2, z2)

def calc_hidden_error(little_delta_k, z, w, fprime=vsigmoid_derivative):
    return numpy.multiply(little_delta_k.dot(w.T), fprime(z))

def calc_output_error(t, z, fprime=vsigmoid_derivative):
    return numpy.multiply((t - z), fprime(z))

def calc_weight_deltas(c, little_delta_k, z, w):
    little_delta_array = little_delta_k.A1
    z_array = z.A1

    (num_rows, num_cols) = w.shape
    return numpy.matrix([ 
        [ little_delta_array[col] * z_array[row] for col in range(num_cols) ]
        for row in range(num_rows) ])

def calc_bias_deltas(c, little_delta_k, b):
    return calc_weight_deltas(c,
            little_delta_k,
            numpy.matrix(numpy.ones(b.shape)),
            b)
