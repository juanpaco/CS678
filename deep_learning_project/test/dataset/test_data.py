import pytest
import random

from dataset import (load_iris, load_mnist, partition_list, split_list)

def test_load_iris():
    data = load_iris()

    # Did we get the dimensions right?
    assert data['num_inputs'] == 4
    assert data['num_outputs'] == 3

    assert data['data'][0]['input'].shape == (1,4)
    # Were the target values correct?
    # The iris dataset has categorical output, with 3 different classes
    assert data['data'][0]['output'].shape == (1,3)

    # The first kind of iris should be on
    assert data['data'][0]['output'].item((0,0)) == 1
    assert data['data'][0]['output'].item((0,1)) == 0
    assert data['data'][0]['output'].item((0,2)) == 0

    assert data['partitions']['training']

def test_partition_data():
    # Seed it so we get the same results each time
    random.seed(0)

    l = range(0, 100)

    partitioned = partition_list(l)

    assert len(partitioned['training']) == 70
    assert len(partitioned['validation']) == 20
    assert len(partitioned['test']) == 10
   
def test_split_list():
    l = range(0, 100)

    (first, second) = split_list(l, .7)

    assert len(first) == 70
    assert len(second) == 30

def test_load_test_mnist():
    mnist = load_mnist()

    assert mnist['num_inputs'] == 28 * 28
    assert mnist['num_outputs'] == 10

    assert mnist['partitions']['test'][0] == 60000
    assert max(mnist['partitions']['test']) == 69999

    print('train len', len(mnist['partitions']['training']))

    assert mnist['data'][0]['output'].A[0].tolist().index(1) == 5
    assert mnist['data'][1]['output'].A[0].tolist().index(1) == 0
    assert mnist['data'][2]['output'].A[0].tolist().index(1) == 4
    assert mnist['data'][3]['output'].A[0].tolist().index(1) == 1

