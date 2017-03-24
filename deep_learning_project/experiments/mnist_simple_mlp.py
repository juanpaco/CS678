from dataset import (load_mnist)
from model import (
        evaluate_net,
        random_weights,
        train)

print('load dataset')
mnist = load_mnist()

print('initialize with random weights')
net = random_weights(mnist, [ 100 ])

print('Random net %: ', evaluate_net(mnist, net, 'test') * 100)

print('begin training')
res = train(mnist, net, .1, 1000)

print('Trained net %: ', evaluate_net(mnist, net, 'test') * 100)
