import numpy
from anchor_words import (process_dataset)
from read_dataset import (tokenize_dataset)

print('Process clinton')
random = numpy.random.RandomState(1)
raw_data = tokenize_dataset('clinton')

process_dataset(raw_data, random)
