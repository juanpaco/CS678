from collections import deque
from functools import (reduce)
import numpy

from .net import(feed_forward, init_net, vsigmoid, vsigmoid_derivative)

def calc_hidden_error(little_delta_k, z, w, fprime=vsigmoid_derivative):
    return numpy.multiply(little_delta_k.dot(w.T), fprime(z))

def calc_output_error(t, z, fprime=vsigmoid_derivative):
    return numpy.multiply((t - z), fprime(z))

def calc_weight_deltas(c, errors, i, zs, net):
    deltas = []

    # Align the signals with the net weights
    signals = deque(zs)
    signals.appendleft(i)
    signals.pop()

    for i in range(0, len(errors)):
        w = net[i]['w']
        (num_rows, num_cols) = w.shape
        delta_array = errors[i].A1
        z_array = signals[i].A1

        w_deltas = numpy.matrix([ 
            [ c * delta_array[col] * z_array[row] for col in range(num_cols) ]
            for row in range(num_rows) ])

        b_signal = numpy.matrix(numpy.ones((1, num_cols)))
        b_deltas = numpy.matrix([[ c * delta_array[col] for col in range(num_cols) ]])

        deltas.append({ 'w': w_deltas, 'b': b_deltas })

    return deltas

# returns a list of little deltas
def compute_errors(t, zs, net):
    errors = []

    for index in range(len(net) - 1, -1, -1):
        if len(errors) == 0:
            errors.append(calc_output_error(t, zs[index]))
        else:
            errors.append(calc_hidden_error(errors[-1], zs[index], net[index+1]['w'])) 

    return list(reversed(errors))

# c i learning rate
# i - input
# net - array of weights and biases
def backprop_iteration(c, i, net, t, activation=vsigmoid):
    """ output would be the updated weights """

    #print('c', c)
    #print('i', i)
    #print('net', net)
    #print('t', t)

    zs = feed_forward(i, net, activation)
    errors = compute_errors(t, zs, net)
    weight_deltas = calc_weight_deltas(c, errors, i, zs, net)

    foo = list(zip(weight_deltas, net))

    return [ { 'w': numpy.add(i[0]['w'], i[1]['w']), 'b': numpy.add(i[0]['b'], i[1]['b']) }
        for i in zip(weight_deltas, net) ]

def epoch(dataset, net, c):
    def tick(net, input_index):
        return backprop_iteration(c,
            dataset['data'][input_index]['input'],
            net,
            dataset['data'][input_index]['output'])

    return reduce(tick, dataset['partitions']['training'], net)

def evaluate_net(dataset, net):
    def coerce_output(output):
        return [ 1 if i == numpy.argmax(output) else 0
                for i in range(0, output.shape[1]) ]

    def compare(actual, target):
        coerced_actual = coerce_output(actual)

        #print(coerced_actual, target.A[0])

        return 1 if numpy.array_equal(coerced_actual, target.A[0]) else 0

    def reduce_count(count, ind):
        zs = feed_forward(dataset['data'][ind]['input'], net)

        return count + compare(zs[-1], dataset['data'][ind]['output'])

    correct_count = reduce(reduce_count, dataset['partitions']['validation'], 0)

    return correct_count / len(dataset['partitions']['validation'])

def train(dataset, layer_sizes, c):
    epochs = 0

    net = init_net(dataset, layer_sizes)
    print('count_correct', evaluate_net(dataset, net))

    for i in range(0,1000):
      net = epoch(dataset, net, c)
      print(i, ': validation %: ', evaluate_net(dataset, net) * 100)

    return net
