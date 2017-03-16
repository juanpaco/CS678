# This file has functions for bootstrapping a new net
import numpy

def random_weights(dataset, layer_sizes):
    """ Initializes a network with random weights """

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

     

