import numpy
from anchor_words import (process_dataset)

print('Process sotu')
random = numpy.random.RandomState(1)

process_dataset('sotu', random, inclusion_threshold=2)
