import numpy
from anchor_words import (process_dataset)
from read_dataset import (tokenize_dataset)

print('Process sotu')
random = numpy.random.RandomState(1)

process_dataset('sotu', random)
