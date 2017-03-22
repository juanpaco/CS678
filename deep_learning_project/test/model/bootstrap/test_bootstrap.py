import numpy
import random
import pytest

from dataset import (load_iris)
from model.bootstrap import (
        dataset_to_sac_dataset,
        make_prep_sac_layer,
        random_weights,
        stacked_auto_encoder,
        )

def test_init_net():
    dataset = load_iris()

    net = random_weights(dataset, [ 20 ])

    assert net[0][0].shape == (4, 20)
    assert net[0][1].shape == (1, 20)

    assert net[1][0].shape == (20, 3)
    assert net[1][1].shape == (1, 3)

#def test_dataset_to_sac_dataset_with_no_current_net():
#    random.seed(0)
#
#    dataset = load_iris()
#
#    new_dataset = dataset_to_sac_dataset(dataset, [])
#
#    new_dataset_length = len(new_dataset['data'])
#    
#    assert new_dataset_length == len(new_dataset['data'])

def test_dataset_to_sac_dataset_with_a_net():
    # make up some net and pass it in.
    # with that net I should get a known value for the first training instance
    # so build the test around getting that
    random.seed(0)

    instances = [
            { 'input': numpy.matrix('1 2 3'), 'output': numpy.matrix('5') },
            { 'input': numpy.matrix('4 5 6'), 'output': numpy.matrix('5') },
            ]

    dataset = {
            'data': instances,
            'num_inputs': 3,
            'num_outputs': 1,
            'partitions': {} # don't actually need them for this test
            }

    layer = (
            numpy.matrix('.5 .5 .5; .5 .5 .5; .5 .5 .5'),
            numpy.matrix('.1 .1 .1'),
            )

    current_net = [ layer ]

    new_dataset = dataset_to_sac_dataset(dataset, current_net)

    new_dataset_length = len(new_dataset['partitions']['training'])
    first_instance = new_dataset['data'][0]['input']
    first_output = new_dataset['data'][0]['output']

    assert new_dataset['num_outputs'] == dataset['num_inputs']

    assert new_dataset_length == len(new_dataset['data'])

    assert first_instance.A[0][0] == pytest.approx(0.95689275, abs=.00001)
    assert first_instance.A[0][1] == pytest.approx(0.95689275, abs=.00001)
    assert first_instance.A[0][2] == pytest.approx(0.95689275, abs=.00001)

    assert first_instance.A[0][0] == first_output.A[0][0]
    assert first_instance.A[0][1] == first_output.A[0][1]
    assert first_instance.A[0][2] == first_output.A[0][2]

def test_stacked_auto_encoder():
    random.seed(0)
    numpy.random.seed(0)

    instances = [
            { 'input': numpy.matrix('1 2 3'), 'output': numpy.matrix('5') },
            ]

    dataset = {
            'data': instances,
            'num_inputs': 3,
            'num_outputs': 1,
            'partitions': {} # don't actually need them for this test
            }

    initial_net = stacked_auto_encoder(dataset, [ 2, 2 ], .1, 1)

    w1 = initial_net[0][0].A
    b1 = initial_net[0][1].A

    assert w1[0][0] == pytest.approx(0.02688863491, abs=.000001)
    assert w1[0][1] == pytest.approx(0.1110200495, abs=.000001)
    assert w1[1][0] == pytest.approx(0.05634545982, abs=.000001)
    assert w1[1][1] == pytest.approx(0.02929232898, abs=.000001)
    assert w1[2][0] == pytest.approx(-0.03072694527, abs=.000001)
    assert w1[2][1] == pytest.approx(0.08322316847, abs=.000001)
    assert b1[0][0] == pytest.approx(-0.02872450509, abs=.000001)
    assert b1[0][1] == pytest.approx(0.1993118695, abs=.000001)

    w2 = initial_net[1][0].A
    b2 = initial_net[1][1].A

    assert w2[0][0] == pytest.approx(0.16616113, abs=.000001)
    assert w2[0][1] == pytest.approx(0.13895135, abs=.000001)
    assert w2[1][0] == pytest.approx(0.18481446, abs=.000001)
    assert w2[1][1] == pytest.approx(0.23914559, abs=.000001)
    assert b2[0][0] == pytest.approx(0.14928445, abs=.000001)
    assert b2[0][1] == pytest.approx(-0.01951202, abs=.000001)
    print('not done yet')

def test_prep_sac_layer():
    random.seed(0)
    numpy.random.seed(0)

    instances = [
            { 'input': numpy.matrix('1 2'), 'output': numpy.matrix('5') },
            ]

    dataset = {
            'data': instances,
            'num_inputs': 2,
            'num_outputs': 1,
            'partitions': {} # don't actually need them for this test
            }

    layer = (
            numpy.matrix('.5 .5; .5 .5'),
            numpy.matrix('.1 .1'),
            )

    current_net = [ layer ]

    new_net = make_prep_sac_layer(dataset, .1, 1)(current_net, 3)

    # we had a net with only 1 layer.  It should be 2 now.
    assert len(new_net) == 2

    weights = new_net[1][0].A

    assert weights[0][0] == pytest.approx(0.02459622524, abs=.000001)
    assert weights[0][1] == pytest.approx(0.1076893874, abs=.000001)
    assert weights[0][2] == pytest.approx(0.05134015041, abs=.000001)
    assert weights[1][0] == pytest.approx(0.02263106524, abs=.000001)
    assert weights[1][1] == pytest.approx(-0.03807789256, abs=.000001)
    assert weights[1][2] == pytest.approx(0.07290552041, abs=.000001)

