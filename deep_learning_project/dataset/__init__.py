from mnist import MNIST
import csv
import math
import numpy
from os.path import join
import random

data_root = 'data'

def load(dataset):
    path = join(data_root, dataset + '.data')

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        lines = [ row for row in reader ]

    return lines

def load_iris(train_percentage=.7, validation_percentage=.2):
    data = load('iris')

    output_map = { 'Iris-setosa': [ 1, 0, 0 ],
            'Iris-versicolor': [ 0, 1, 0 ],
            'Iris-virginica': [ 0, 0, 1 ]}

    def build_instance(row):
        input = list(map(float, row[0:4]))
        output = output_map[row[4]]

        return { 'input': numpy.matrix([input]),
                'output': numpy.matrix([output]) }

    instances = list(map(build_instance, data))

    return { 'data': instances,
            'num_inputs': 4,
            'num_outputs': 3,
            'partitions': partition_list(instances)}

# This one partitions the 3 sets all off of the original data.  I've sometimes
#   seen people split training and testing, and then split validation out of
#   training.  Meh.
def partition_list(l, train_percentage=.7, validation_percentage=.2):
    indices = list(range(0, len(l)))
    random.shuffle(indices)

    train_indices = (0, math.floor(train_percentage * len(l)))
    validation_indices = (train_indices[1],
            train_indices[1] + math.floor(validation_percentage * len(l)))
    test_indices = (validation_indices[1], len(l))

    return {
            'training': indices[slice(*train_indices)],
            'validation': indices[slice(*validation_indices)],
            'test': indices[slice(*test_indices)]}

# This one just splits the list in 2.  First percentage should actually be the
#   decimal number corresponding to the desired percentage.  So, if you want
#   70% in the first group, pass in .7.
def split_list(l, first_percentage):
    indices = list(range(0, len(l)))
    random.shuffle(indices)

    first_indices = (0, math.floor(first_percentage * len(l)))
    second_indices = (first_indices[1], len(l))

    first = indices[slice(*first_indices)]
    second = indices[slice(*second_indices)]

    return (first, second)

def load_mnist():
    mndata = MNIST('./data')

    training = mndata.load_training()
    testing = mndata.load_testing()

    output_map = [
        numpy.matrix([ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]),
        numpy.matrix([ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ]),
        numpy.matrix([ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ]),
        numpy.matrix([ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]),
        numpy.matrix([ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ]),
        numpy.matrix([ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ]),
        numpy.matrix([ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ]),
        numpy.matrix([ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ]),
        numpy.matrix([ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]),
        numpy.matrix([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ]),
    ]

    training_instances = [
        {
            'input': numpy.matrix(training[0][i]) / 255,
            'output': output_map[training[1][i]],
        } for i in range(0, len(training[0])) ]

    test_instances = [
        {
            'input': numpy.matrix(testing[0][i]) / 255,
            'output': output_map[testing[1][i]],
        } for i in range(0, len(testing[0])) ]

    (training_indices, validation_indices) = split_list(
            training_instances,
            .8,
        )

    all_data = training_instances + test_instances

    return {
        'data': all_data,
        'num_inputs': 784, # It just is with MNIST
        'num_outputs': 10,
        'partitions': {
            'training': training_indices,
            'validation': validation_indices,
            'test': list(range(len(training_instances), len(all_data))),
            },
        }
