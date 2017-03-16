from dataset import (load_iris)

from model import (random_weights)

def test_init_net():
    dataset = load_iris()

    net = random_weights(dataset, [ 20 ])

    assert net[0]['w'].shape == (4, 20)
    assert net[0]['b'].shape == (1, 20)

    assert net[1]['w'].shape == (20, 3)
    assert net[1]['b'].shape == (1, 3)
