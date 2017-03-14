import numpy
import pytest

from dataset import (load_iris)
from model.net import (compute_layer,
        feed_forward,
        init_net,
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

def test_init_net():
    dataset = load_iris()

    net = init_net(dataset, [ 20 ])

    assert net[0]['w'].shape == (4, 20)
    assert net[0]['b'].shape == (1, 20)

    assert net[1]['w'].shape == (20, 3)
    assert net[1]['b'].shape == (1, 3)
