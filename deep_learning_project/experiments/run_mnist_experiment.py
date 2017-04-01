import numpy
import random
import time

from dataset import (load_mnist)
from experiments.options import (get_settings)
from model import (
        evaluate_net,
        random_weights,
        stacked_auto_encoder,
        train)

def run(shape, learning_rate=.1, corruption_rate=0, decay_rate=0, hidden_epochs=100, seed=0, init_with='random'):
    random.seed(seed)
    numpy.random.seed(seed)

    start_time = time.time()
    
    print('Running mnist with:',)
    print('\tshape:', shape)
    print('\tlearning_rate:', learning_rate)
    print('\tcorruption_rate:', corruption_rate)
    print('\tdecay_rate:', decay_rate)
    print('\thidden_epochs:', hidden_epochs)
    print('\tseed:', seed)
    print('\tinit_with:', init_with)

    print('load dataset')
    mnist = load_mnist()

    data_loaded_at_time = time.time()
    
    if init_with == 'sac':
        print('initialize with sac')
        initial_net = stacked_auto_encoder(
                mnist,
                shape,
                learning_rate,
                hidden_epochs=hidden_epochs,
                corruption_rate=corruption_rate,
                decay_rate=decay_rate,
            )
    else:
        initial_net = random_weights(mnist, shape)

    
    print('Initial net %: ', evaluate_net(mnist, initial_net, 'test') * 100)

    initialization_done_time = time.time()
    
    # Now train it on the actual data
    refined_net = train(
            mnist,
            initial_net,
            learning_rate,
            # This is the max number of epochs to allow.  With patience I've
            #   never come close to hitting this.
            1000, 
        )
    
    print('After refinement %: ', evaluate_net(mnist, refined_net, 'test') * 100)

    end_time = time.time()

    print('start time:', start_time)
    print('data done loading:', data_loaded_at_time)
    print('init done at:', initialization_done_time)
    print('end time:', end_time)
    print('total elapsed time:', end_time - start_time)

if __name__ == "__main__":
    settings = get_settings()

    run(
            settings['shape'],
            settings['learning-rate'],
            settings['corruption-rate'],
            settings['decay-rate'],
            settings['hidden-epochs'],
            settings['seed'],
            settings['init-with'],
        )
