from collections import deque
from functools import (reduce)
import numpy
import random

from .net import(feed_forward, vsigmoid, vsigmoid_derivative)

def decay(weight, rate):
    return weight * rate

decayv = numpy.vectorize(decay)

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
def backprop_iteration(c, i, net, t, decay_rate=0, activation=vsigmoid):
    """ output would be the updated weights """

    #print('c', c)
    #print('i', i)
    #print('net', net)
    #print('t', t)
    #print('decay_rate', decay_rate)

    zs = feed_forward(i, net, activation)
    errors = compute_errors(t, zs, net)
    weight_deltas = calc_weight_deltas(c, errors, i, zs, net)

    updated_weights = decay_weights(net, decay_rate)

    updated_weights = [
            ( numpy.add(i[0][0], i[1][0]), numpy.add(i[0][1], i[1][1]) ) 
            for i in zip(weight_deltas, updated_weights)
        ]


    return updated_weights

# net is the net prior to receiving weight updates
# rate is between 0 and 1
# returns the net with its weights decayed by decay_rate
def decay_weights(net, rate):
    if rate == 0:
        return net

    decay_deltas  = [ ( decayv(i[0], rate), decayv(i[1], rate) ) for i in net ]

    return [
            ( numpy.subtract(i[0][0], i[1][0]), numpy.subtract(i[0][1], i[1][1]) ) 
            for i in zip(net, decay_deltas)
        ]

def epoch(dataset, net, c, corruption_rate, decay_rate):
    current_net = net
#    count = 0

    for i in dataset['partitions']['training']:
        if corruption_rate > 0:
            the_input = corrupt_input(dataset['data'][i]['input'], corruption_rate)
        else:
            the_input = dataset['data'][i]['input']

        #print('original', dataset['data'][i]['input'])
        #print('corrupted', the_input)
#        count += 1
#        print('count:', count)
        current_net = backprop_iteration(c,
            the_input,
            current_net,
            dataset['data'][i]['output'],
            decay_rate=decay_rate,
            )

    return current_net

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

def corrupt_input(i, corruption_rate):
    new_input = i.copy()

    for x in range(0, i.size):
        if random.uniform(0,1) <= corruption_rate:
            new_input.itemset(x, 0)

    return new_input


def train(dataset, net, c, epochs, corruption_rate=0, decay_rate=0, patience=20, evaluate=True):
    print('Start training:')
    print('\ttraining instances:', len(dataset['partitions']['training']))
    print('\tvalidation instances:', len(dataset['partitions']['validation']))
    print('\ttest instances:', len(dataset['partitions']['test']))
    print('\tpatience:', patience)
    print('\tmax epocs:', epochs)
    print('\tcorruption rate:', corruption_rate)
    print('\tdecay rate:', decay_rate)

    best_net = None
    best_net_score = 0
    iterations_since_improvment = 0
    current_net = net

    for i in range(0, epochs):
        #if i % 1000 == 0:
        #    print('epoch:', i)
        current_net = epoch(dataset, current_net, c, corruption_rate, decay_rate)

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
