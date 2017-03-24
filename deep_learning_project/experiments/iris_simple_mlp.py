from dataset import (load_iris)
from model import (
        evaluate_net,
        random_weights,
        train)

print('load dataset')
dataset = load_iris()

print('initialize with random weights')
net = random_weights(dataset, [ 4 ])

print('Random net %: ', evaluate_net(dataset, net, 'test') * 100)

print('begin training')
res = train(dataset, net, .1, 1000)

print('Trained net %: ', evaluate_net(dataset, res, 'test') * 100)
