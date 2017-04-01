# This file has functions for bootstrapping a new net
from functools import (reduce)
import numpy
import random

from .net import (feed_forward)
from .train import (train)

def random_weights(dataset, layer_sizes):
    """ Initializes a network with random weights """

    complete_layer_sizes = ([dataset['num_inputs']] +
        layer_sizes +
        [dataset['num_outputs']])

    def init_layer(upper_index):
        weight_shape = (complete_layer_sizes[upper_index-1],
                complete_layer_sizes[upper_index])
        bias_shape = (1, complete_layer_sizes[upper_index])

        return (
                numpy.matrix((numpy.random.random(weight_shape) - .5) / 2),
                numpy.matrix((numpy.random.random(bias_shape) -.5) / 2) 
               )

    return [ init_layer(i) for i in range(1, len(complete_layer_sizes)) ]

def stacked_auto_encoder(dataset, layer_sizes, c, hidden_epochs=100, corruption_rate=0, decay_rate=0):
    final_net = []

    prep_sac_layer = make_prep_sac_layer(dataset, c, hidden_epochs, corruption_rate, decay_rate)

    final_net = reduce(
            prep_sac_layer,
            layer_sizes,
            []
        )

    return final_net

# As I'm initializing the hidden layer weights, I'm building up a net.  For
#   layers other than the 1st one, the dataset inputs aren't the actualy inputs
#   I need for the training.  They actually need the output from the net for the
#   layers leading up to them.  I'm choosing to construct a new dataset that is
#   the result of pumping the original dataset through the net up to that point.
#   I don't think my datasets will be so large that I'll run out of memory, and
#   feeding forward through the net over and over again seems like it'll take a
#   lot of time.  I'm choosing time efficiency over space efficiency.
def dataset_to_sac_dataset(dataset, current_net):
    print('dataset_to_sac_dataset:', len(current_net))

    indices = list(range(0, len(dataset['data'])))
    random.shuffle(indices)

    new_instances = [ process_sac_input(dataset['data'][i], current_net)
            for i in indices ]

    new_size = new_instances[0]['input'].shape[1]

    return { 'data': new_instances,
        'num_inputs': new_size,
        'num_outputs': new_size,
        'partitions': { 'training': indices, 'validation': [], 'test': [] }
        }

def process_sac_input(i, current_net):
    if len(current_net) == 0:
        z = i['input']
    else:
        z = feed_forward(i['input'], current_net)[-1]

    new_input = numpy.matrix(z)

    return { 'input': new_input, 'output': new_input }

def make_prep_sac_layer(dataset, c, epochs, corruption_rate, decay_rate):
    def prep_sac_layer(layers, layer_size):

        prepped_dataset = dataset_to_sac_dataset(dataset, layers)

        net = random_weights(prepped_dataset, [ layer_size ])
        print('Initialized random net')

        new_net = train(
                prepped_dataset,
                net,
                c,
                epochs,
                corruption_rate,
                decay_rate=decay_rate,
                evaluate=False
            )

        layers.append(new_net[0])
        
        print('layer done')
        return layers

    return prep_sac_layer
