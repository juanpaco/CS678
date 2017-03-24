import numpy
import random

from dataset import (load_mnist)
from model import (
        evaluate_net,
        stacked_auto_encoder,
        train)

# I'm going to be comparing this to fine-tuning and regularization later. I
#   want an apples-to-apples comparison.  Determinism ftw!
random.seed(0)
numpy.random.seed(0)

print('load dataset')
mnist = load_mnist()

print('initialize with sac')
initial_net = stacked_auto_encoder(mnist, [ 100, 50, 25, 10 ], .1, 100)

print('Sac only net %: ', evaluate_net(mnist, initial_net, 'test') * 100)
