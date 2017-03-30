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
        corrupt_input,
        evaluate_net,
        feed_forward,
        random_weights,
        stacked_auto_encoder,
        train)

# This also uses the network from HW1
def test_calc_errors():
    i = numpy.matrix('0.4 0.9')

    net = [
            ( numpy.matrix('1 1.2; 0.5 0.5'), numpy.matrix('0 0.5') ),
            ( numpy.matrix('0.1; -0.8'), numpy.matrix('-1.3') )
        ]

    t = numpy.matrix('0.1')

    (z1, z2) = feed_forward(i, net)

    output_error = calc_output_error(t, z2)
    hidden_error = calc_hidden_error(output_error, z1, net[1][0])

    assert output_error.item((0,0)) == pytest.approx(-0.0037, abs=.0001)

    assert hidden_error.item((0,0)) == pytest.approx(-0.000079, abs=.00001)
    assert hidden_error.item((0,1)) == pytest.approx(0.000469, abs=.0001)

def test_calc_weight_deltas():
    learning_rate = .1

    i = numpy.matrix('0.4 0.9')

    net = [
            ( numpy.matrix('1 1.2; 0.5 0.5'), numpy.matrix('0 0.5') ),
            ( numpy.matrix('0.1; -0.8'), numpy.matrix('-1.3') )
        ]

    t = numpy.matrix('0.1')

    zs = feed_forward(i, net)

    errors = compute_errors(t, zs, net)
    deltas = calc_weight_deltas(learning_rate, errors, i, zs, net)

    assert deltas[0][1].shape == (1,2)

    assert deltas[1][0].item((0,0)) == pytest.approx(-.000265, abs=.00001)
    assert deltas[1][0].item((1,0)) == pytest.approx(-.000306, abs=.00001)

    assert deltas[1][1].item((0,0)) == pytest.approx(-0.000379, abs=.00001)
    assert deltas[1][1].shape == (1,1)

    new_w2 = numpy.add(net[1][0], deltas[1][0])

    assert new_w2.item((0,0)) == pytest.approx(0.09973, abs=.0001)

def test_backprop_iteration():
    # learning rate
    c = 1

    i = numpy.matrix('0.4 0.9')

    net = [
            ( numpy.matrix('1 1.2; 0.5 0.5'), numpy.matrix('0 0.5') ),
            ( numpy.matrix('0.1; -0.8'), numpy.matrix('-1.3') )
          ]

    t = numpy.matrix('0.1')

    new_net = backprop_iteration(c, i, net, t)

    assert new_net[0][0].item((0,0)) == pytest.approx(0.999968, abs=.00001)

def test_compute_errors():
    i = numpy.matrix('0.4 0.9')
    t = numpy.matrix('0.1')

    net = [
            ( numpy.matrix('1 1.2; 0.5 0.5'), numpy.matrix('0 0.5') ),
            ( numpy.matrix('0.1; -0.8'), numpy.matrix('-1.3') )
          ]

    zs = feed_forward(i, net)

    errors = compute_errors(t, zs, net)

    assert errors[1].item((0,0)) == pytest.approx(-0.0037, abs=.0001)

    assert errors[0].item((0,0)) == pytest.approx(-0.000079, abs=.00001)
    assert errors[0].item((0,1)) == pytest.approx(0.000469, abs=.0001)

def test_corrupt_input():
    random.seed(1)

    i = numpy.matrix('1 2 3 4 5 6 7 8 9 10')

    corrupted_input = corrupt_input(i, .3)

    assert corrupted_input.item((0,0))  == 0
    assert corrupted_input.item((0,1))  == 2
    assert corrupted_input.item((0,2))  == 3
    assert corrupted_input.item((0,3))  == 0
    assert corrupted_input.item((0,4))  == 5
    assert corrupted_input.item((0,5))  == 6
    assert corrupted_input.item((0,6))  == 7
    assert corrupted_input.item((0,7))  == 8
    assert corrupted_input.item((0,8))  == 0
    assert corrupted_input.item((0,9))  == 0
