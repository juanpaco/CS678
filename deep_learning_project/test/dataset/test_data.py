import pytest
import random

from dataset import (load_iris, partition_list)

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
   
