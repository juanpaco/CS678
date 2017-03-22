import numpy
import pytest

from model import (compute_layer, feed_forward)
from model.net import(sigmoid)

def test_sigmoid():
    assert sigmoid(0) == .5

def test_compute_layer():
    i = numpy.matrix('1 2')

    layer = (
            numpy.matrix('.5 .5; .5 .5'),
            numpy.matrix('.1 .1')
        )

    [ z ] = compute_layer(i, layer)

    assert z.item((0,0)) == pytest.approx(.832, abs=.001)
    assert z.item((0,1)) == pytest.approx(.832, abs=.001)

def test_feed_forward():
    i = numpy.matrix('1 2')

    net = [
            ( numpy.matrix('.5 .5; .5 .5'), numpy.matrix('.1 .1') ),
            ( numpy.matrix('.25 .75; .75 .25'), numpy.matrix('.2 .2') )
          ]

    [ z1, z2 ] = feed_forward(i, net)

    assert z2.item((0,0)) == pytest.approx(.737, abs=.001)
    assert z2.item((0,1)) == pytest.approx(.737, abs=.001)

# This uses the network from HW1
def test_another_forward():
    i = numpy.matrix('0.4 0.9')

    net = [
            ( numpy.matrix('1 1.2; 0.5 0.5'), numpy.matrix('0 0.5') ),
            ( numpy.matrix('0.1; -0.8'),  numpy.matrix('-1.3') )
        ]

    (z1, z2) = feed_forward(i, net)

    assert z2.item((0,0)) == pytest.approx(0.133, abs=.001)

    assert z1.item((0,0)) == pytest.approx(0.700, abs=.001)
    assert z1.item((0,1)) == pytest.approx(0.807, abs=.001)
