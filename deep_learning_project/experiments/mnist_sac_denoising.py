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

print('SAC with denoising')
print('load dataset')
mnist = load_mnist()

print('initialize with sac')
initial_net = stacked_auto_encoder(mnist, [ 100, 50, 25, 10 ], .1, 10, corruption_rate=.3)

print('Sac only net %: ', evaluate_net(mnist, initial_net, 'test') * 100)

# Now train it on the actual data
refined_net = train(mnist, initial_net, .1, 1000)

print('After refinement %: ', evaluate_net(mnist, refined_net, 'test') * 100)
