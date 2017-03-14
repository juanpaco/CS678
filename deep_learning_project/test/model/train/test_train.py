import numpy
import pytest

from model import (calc_bias_deltas,
        backprop_iteration,
        calc_hidden_error,
        calc_output_error,
        calc_weight_deltas,
        feed_forward)

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
