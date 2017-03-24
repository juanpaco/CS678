from collections import deque
from functools import (reduce)
import numpy

from .net import(feed_forward, vsigmoid, vsigmoid_derivative)

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
        w = net[i][0]
        (num_rows, num_cols) = w.shape
        delta_array = errors[i].A1
        z_array = signals[i].A1

        w_deltas = numpy.matrix([ 
            [ c * delta_array[col] * z_array[row] for col in range(num_cols) ]
            for row in range(num_rows) ])

        b_signal = numpy.matrix(numpy.ones((1, num_cols)))
        b_deltas = numpy.matrix([[ c * delta_array[col] for col in range(num_cols) ]])

        deltas.append(( w_deltas, b_deltas ))

    return deltas

# returns a list of little deltas
def compute_errors(t, zs, net):
    errors = []

    for index in range(len(net) - 1, -1, -1):
        if len(errors) == 0:
            errors.append(calc_output_error(t, zs[index]))
        else:
            errors.append(calc_hidden_error(errors[-1], zs[index], net[index+1][0])) 

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

    return [ ( numpy.add(i[0][0], i[1][0]), numpy.add(i[0][1], i[1][1]) ) 
        for i in zip(weight_deltas, net) ]

def epoch(dataset, net, c):
    current_net = net
#    count = 0

    for i in dataset['partitions']['training']:
#        count += 1
#        print('count:', count)
        current_net = backprop_iteration(c,
            dataset['data'][i]['input'],
            current_net,
            dataset['data'][i]['output'])

    return current_net

    #def tick(net, input_index):
    #    #print ('tick:')
    #    #print('input', dataset['data'][input_index]['input'], dataset['data'][input_index]['output'], net)
    #    return backprop_iteration(c,
    #        dataset['data'][input_index]['input'],
    #        net,
    #        dataset['data'][input_index]['output'])

    #return reduce(tick, dataset['partitions']['training'], net)

# against_set - a string, one of 'validation' or 'test'
def evaluate_net(dataset, net, against_set):
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

    correct_count = reduce(reduce_count, dataset['partitions'][against_set], 0)

    return correct_count / len(dataset['partitions'][against_set])

def train(dataset, net, c, epochs, patience=20, evaluate=True):
    print('Start training:')
    print('\ttraining instances:', len(dataset['partitions']['training']))
    print('\tvalidation instances:', len(dataset['partitions']['validation']))
    print('\ttest instances:', len(dataset['partitions']['test']))
    print('\tpatience:', patience)

    best_net = None
    best_net_score = 0
    iterations_since_improvment = 0
    current_net = net

    for i in range(0, epochs):
        #if i % 1000 == 0:
        #    print('epoch:', i)
        current_net = epoch(dataset, current_net, c)

        if evaluate:
            validation_score = evaluate_net(dataset, current_net, 'validation')
            print(
                i,
                ': validation %: ',
                validation_score * 100
                )

            if (validation_score > best_net_score):
                best_net_score = validation_score
                best_net = current_net
                iterations_since_improvment = 0
            else:
                iterations_since_improvment += 1

            if iterations_since_improvment == patience:
                current_net = best_net
                print('**Breaking because we passed patience threshold')
                break

    return current_net 
