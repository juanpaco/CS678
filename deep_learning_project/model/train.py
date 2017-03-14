import numpy

from .net import(feed_forward, vsigmoid, vsigmoid_derivative)

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

# c i learning rate
# i - input
# w1 - weights between input and hidden
# b1 - bias to hidden
# w2 - weights between hidden and output
# b2 - bias to output
def backprop_iteration(c, i, w1, b1, w2, b2, t, activation=vsigmoid):
    """ output would be the updated weights """

    (z1, z2) = feed_forward(i, w1, b1, w2, b2, activation)

    # TODO: Should get the derivative function passed in here, but for right now
    #   we only pretend that it's configurable.
    output_error = calc_output_error(t, z2, vsigmoid_derivative)
    hidden_error = calc_hidden_error(output_error, z1, w2)

    w2_deltas = calc_weight_deltas(c, output_error, z1, w2)
    b2_deltas = calc_bias_deltas(c, output_error, b2)

    w1_deltas = calc_weight_deltas(c, hidden_error, i, w1)
    b1_deltas = calc_bias_deltas(c, hidden_error, b1)

    return (w1 + w1_deltas,
            b1 + b1_deltas,
            w2 + w2_deltas,
            b2 + b2_deltas)
