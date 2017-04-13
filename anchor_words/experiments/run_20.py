import numpy
from anchor_words import (process_dataset)
from read_dataset import (tokenize_dataset)

print('Process 20')
random = numpy.random.RandomState(1)
raw_data = tokenize_dataset('20')

process_dataset(raw_data, random)
