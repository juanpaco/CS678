import numpy
from anchor_words import (process_dataset)

print('Process clinton')
random = numpy.random.RandomState(1)

process_dataset('clinton', random, inclusion_threshold=2)
