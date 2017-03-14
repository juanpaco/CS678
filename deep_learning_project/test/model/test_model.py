import numpy
import pytest

from dataset import (load_iris)
from model import (calc_bias_deltas,
        backprop_iteration,
        calc_hidden_error,
        calc_output_error,
        calc_weight_deltas, 
        compute_layer,
        feed_forward,
        sigmoid)

def test_sigmoid():
    assert sigmoid(0) == .5

def test_compute_layer():
    i = numpy.matrix('1 2')
    w = numpy.matrix('.5 .5; .5 .5')
    b = numpy.matrix('.1 .1')

    z = compute_layer(i, w, b)

    assert z.item((0,0)) == pytest.approx(.832, abs=.001)
    assert z.item((0,1)) == pytest.approx(.832, abs=.001)

def test_feed_forward():
    i = numpy.matrix('1 2')
    w1 = numpy.matrix('.5 .5; .5 .5')
    b1 = numpy.matrix('.1 .1')

    w2 = numpy.matrix('.25 .75; .75 .25')
    b2 = numpy.matrix('.2 .2')

    (z1, z2) = feed_forward(i, w1, b1, w2, b2)

    assert z2.item((0,0)) == pytest.approx(.737, abs=.001)
    assert z2.item((0,1)) == pytest.approx(.737, abs=.001)

# This uses the network from HW1
def test_another_forward():
    i = numpy.matrix('0.4 0.9')
    w1 = numpy.matrix('1 1.2; 0.5 0.5')
    b1 = numpy.matrix('0 0.5')

    w2 = numpy.matrix('0.1; -0.8')
    b2 = numpy.matrix('-1.3')

    (z1, z2) = feed_forward(i, w1, b1, w2, b2)

    assert z2.item((0,0)) == pytest.approx(0.133, abs=.001)

    assert z1.item((0,0)) == pytest.approx(0.700, abs=.001)
    assert z1.item((0,1)) == pytest.approx(0.807, abs=.001)

# This also uses the network from HW1
def test_calc_errors():
    i = numpy.matrix('0.4 0.9')
    w1 = numpy.matrix('1 1.2; 0.5 0.5')
    b1 = numpy.matrix('0 0.5')

    w2 = numpy.matrix('0.1; -0.8')
    b2 = numpy.matrix('-1.3')

    t = numpy.matrix('0.1')

    (z1, z2) = feed_forward(i, w1, b1, w2, b2)

    output_error = calc_output_error(t, z2)
    hidden_error = calc_hidden_error(output_error, z1, w2)

    assert output_error.item((0,0)) == pytest.approx(-0.0037, abs=.0001)

    assert hidden_error.item((0,0)) == pytest.approx(-0.000079, abs=.00001)
    assert hidden_error.item((0,1)) == pytest.approx(0.000469, abs=.0001)

def test_calc_weight_deltas():
    learning_rate = 1

    i = numpy.matrix('0.4 0.9')
    w1 = numpy.matrix('1 1.2; 0.5 0.5')
    b1 = numpy.matrix('0 0.5')

    w2 = numpy.matrix('0.1; -0.8')
    b2 = numpy.matrix('-1.3')

    t = numpy.matrix('0.1')

    (z1, z2) = feed_forward(i, w1, b1, w2, b2)

    output_error = calc_output_error(t, z2)
    hidden_error = calc_hidden_error(output_error, z1, w2)

    w2_deltas = calc_weight_deltas(learning_rate, output_error, z1, w2)
    b2_deltas = calc_bias_deltas(learning_rate, output_error, b2)

    assert w2_deltas.item((0,0)) == pytest.approx(-.00265, abs=.00001)
    assert w2_deltas.item((1,0)) == pytest.approx(-.00306, abs=.00001)

    assert b2_deltas.item((0,0)) == pytest.approx(-0.00379, abs=.00001)

    new_w2 = numpy.add(w2, w2_deltas)

    assert new_w2.item((0,0)) == pytest.approx(0.0973, abs=.0001)

def test_backprop_iteration():
    # learning rate
    c = 1

    i = numpy.matrix('0.4 0.9')
    w1 = numpy.matrix('1 1.2; 0.5 0.5')
    b1 = numpy.matrix('0 0.5')

    w2 = numpy.matrix('0.1; -0.8')
    b2 = numpy.matrix('-1.3')

    t = numpy.matrix('0.1')

    (uw1, ub1, uw2, ub2) = backprop_iteration(c, i, w1, b1, w2, b2, t)

    assert uw1.item((0,0)) == pytest.approx(0.999968, abs=.00001)
    assert ub1.item((0,1)) == pytest.approx(0.500474, abs=.00001)

    assert uw2.item((0,0)) == pytest.approx(0.09734288136, abs=.00001)
    assert ub2.item((0,0)) == pytest.approx(-1.303792811, abs=.00001)

#def test_train_simple():
#    data = 
