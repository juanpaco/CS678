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

