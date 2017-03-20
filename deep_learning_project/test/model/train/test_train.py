import numpy
import pytest
import random

from dataset import (load_iris, load_mnist)
from model import (
        backprop_iteration,
        calc_hidden_error,
        calc_output_error,
        calc_weight_deltas,
        compute_errors,
        evaluate_net,
        feed_forward,
        random_weights,
        stacked_auto_encoder,
        train)

# This also uses the network from HW1
def test_calc_errors():
    i = numpy.matrix('0.4 0.9')

    net = [
            { 'w': numpy.matrix('1 1.2; 0.5 0.5'), 'b': numpy.matrix('0 0.5') },
            { 'w': numpy.matrix('0.1; -0.8'), 'b': numpy.matrix('-1.3') }
        ]

    t = numpy.matrix('0.1')

    (z1, z2) = feed_forward(i, net)

    output_error = calc_output_error(t, z2)
    hidden_error = calc_hidden_error(output_error, z1, net[1]['w'])

    assert output_error.item((0,0)) == pytest.approx(-0.0037, abs=.0001)

    assert hidden_error.item((0,0)) == pytest.approx(-0.000079, abs=.00001)
    assert hidden_error.item((0,1)) == pytest.approx(0.000469, abs=.0001)

def test_calc_weight_deltas():
    learning_rate = .1

    i = numpy.matrix('0.4 0.9')

    net = [
            { 'w': numpy.matrix('1 1.2; 0.5 0.5'), 'b': numpy.matrix('0 0.5') },
            { 'w': numpy.matrix('0.1; -0.8'), 'b': numpy.matrix('-1.3') }
        ]

    t = numpy.matrix('0.1')

    zs = feed_forward(i, net)

    errors = compute_errors(t, zs, net)
    deltas = calc_weight_deltas(learning_rate, errors, i, zs, net)

    assert deltas[0]['b'].shape == (1,2)

    assert deltas[1]['w'].item((0,0)) == pytest.approx(-.000265, abs=.00001)
    assert deltas[1]['w'].item((1,0)) == pytest.approx(-.000306, abs=.00001)

    assert deltas[1]['b'].item((0,0)) == pytest.approx(-0.000379, abs=.00001)
    assert deltas[1]['b'].shape == (1,1)

    new_w2 = numpy.add(net[1]['w'], deltas[1]['w'])

    assert new_w2.item((0,0)) == pytest.approx(0.09973, abs=.0001)

def test_backprop_iteration():
    # learning rate
    c = 1

    i = numpy.matrix('0.4 0.9')

    net = [
            { 'w': numpy.matrix('1 1.2; 0.5 0.5'), 'b': numpy.matrix('0 0.5') },
            { 'w': numpy.matrix('0.1; -0.8'), 'b': numpy.matrix('-1.3') }
          ]

    t = numpy.matrix('0.1')

    new_net = backprop_iteration(c, i, net, t)

    assert new_net[0]['w'].item((0,0)) == pytest.approx(0.999968, abs=.00001)

def test_compute_errors():
    i = numpy.matrix('0.4 0.9')
    t = numpy.matrix('0.1')

    net = [
            { 'w': numpy.matrix('1 1.2; 0.5 0.5'), 'b': numpy.matrix('0 0.5') },
            { 'w': numpy.matrix('0.1; -0.8'), 'b': numpy.matrix('-1.3') }
          ]

    zs = feed_forward(i, net)

    errors = compute_errors(t, zs, net)

    assert errors[1].item((0,0)) == pytest.approx(-0.0037, abs=.0001)

    assert errors[0].item((0,0)) == pytest.approx(-0.000079, abs=.00001)
    assert errors[0].item((0,1)) == pytest.approx(0.000469, abs=.0001)

#def test_train():
#    # Let's always get the same conditions
#    random.seed(0)
#    numpy.random.seed(1)
#
#    dataset = load_iris()
#    net = random_weights(dataset, [ 20 ])
#
#    res = train(dataset, net, .1, 1)
#
#    # We're not actually testing anything here.  Just pass it if it runs.  I
#    #   observed good results.  Can add some specific checks later.
#
#def test_train_auto_encoder():
#    random.seed(0)
#    numpy.random.seed(1)
#
#    dataset = load_iris()
#    net = stacked_auto_encoder(dataset, [ 20, 20, 20 ], .1, 10000)
#    net2 = random_weights(dataset, [ 20 ])
#    net.append(net2[-1])
#
#    print('net', net)
#    trained_net = train(dataset, net, .1, 10000)
#    trained_net2 = train(dataset, net2, .1, 100)
#
#    print(evaluate_net(dataset, net) * 100)
#    print(evaluate_net(dataset, trained_net) * 100)
#    print(evaluate_net(dataset, trained_net2) * 100)
#
#    #print(trained_net)
#    #print(trained_net2)
