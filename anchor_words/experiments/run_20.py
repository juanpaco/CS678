import numpy
from anchor_words import (process_dataset)

print('Process 20')
random = numpy.random.RandomState(1)

process_dataset('20', random, inclusion_threshold=2)
