from functools import reduce 
import math
import numpy
import os.path
import pickle
import scipy

from read_dataset import (tokenize_dataset)

def get_global_vocab_count(documents):
    print('Getting global vocab counts')

    global_vocab_count = {}

    for document in documents:
        for word in document:
            global_vocab_count[word] = global_vocab_count.get(word, 0) + 1

    return global_vocab_count

# A word must occur at least inclusion_threshold times in the corpus before
#   being included in the vocab
def build_vocab_and_wordcounts(documents, inclusion_threshold=1):
    print('building vocab with threshold:', inclusion_threshold)

    global_vocab_count = get_global_vocab_count(documents)
    seen_vocab = {}
    vocab = []
    wordcounts = []

    print('Building vocab_and_word_counts.  Document count:', len(documents))

    # :'( Doing this in an imperative style.  I'm sorry, Haskell!
    for document in documents:
        by_word = {}
        # This is the doc with only words not meeting the threshold
        filtered_doc = []

        for word in document:
            # Don't even consider it if we haven't seen it enough in the vocab
            if global_vocab_count[word] < inclusion_threshold:
                continue

            filtered_doc.append(word)

            if word not in seen_vocab:
                seen_vocab[word] = len(vocab)
                vocab.append(word)

            word_id = seen_vocab[word]
            by_word[word_id] = by_word.get(word_id, 0) + 1

        wordcounts.append({ 'total': len(filtered_doc), 'by_word': by_word })

    original_vocab_size = len(global_vocab_count)
    print('Original vocab size', original_vocab_size)
    print('Working vocab size', len(vocab))
    print('Diff of:', original_vocab_size - len(vocab))

    return { 'vocab': vocab, 'wordcounts': wordcounts }

def build_q(vocab_and_wordcounts):
    vocab = vocab_and_wordcounts['vocab']
    wordcounts = vocab_and_wordcounts['wordcounts']

    h_tilde = scipy.sparse.lil_matrix(
            (len(vocab), len(wordcounts)),
            dtype=float,
        )
    h_hat = numpy.zeros(len(vocab))

    #print(wordcounts)

    for doc_id, wordcount in enumerate(wordcounts):
        doc_length = 0
        normalization_factor = wordcount['total'] * (wordcount['total'] - 1)

        for word_id, count in wordcount['by_word'].items():
            doc_length += count

            # Sometimes the normalization factor comes out as 0.  Don't
            #   normalize if that's the case.
            if normalization_factor > 0:
                h_tilde[word_id, doc_id] = count / numpy.sqrt(normalization_factor)
                h_hat[word_id] += count / normalization_factor

            #print(word_id, ':', count)

    #print('vocab', vocab)
    #print('h_tilde', h_tilde)
    #print('h_hat', h_hat)
    retval = numpy.array(h_tilde * h_tilde.transpose() - numpy.diag(h_hat))
    retval[(-1e-15 < retval) & (retval < 0)] = 0

    return retval

lower_r_bound = 1/6
next_r_bound = 1/3
sqrt_3 = numpy.sqrt(3)
neg_sqrt_3 = -1 * numpy.sqrt(3)

def down_project(matrix, target_dimenstions, random):
    """ Takes matrix of size N X M and projects it to a
    N X <target_dimenstions> matrix.

    Uses the algorithm found in
    https://pdfs.semanticscholar.org/594d/2e123ecb8ec0bc781aec467007d65ab5464d.pdf"""

    def get_value_for_r(whatevs, lolz):
        sample = random.random_sample()

        if (sample < lower_r_bound):
            return sqrt_3
        elif (sample < next_r_bound):
            return neg_sqrt_3
        else:
            return 0.0


    r = numpy.fromfunction(
            numpy.vectorize(get_value_for_r),
            (matrix.shape[1], target_dimenstions),
            dtype=float,
        )

    return numpy.dot(matrix, r)

def normalize_rows(matrix):
    return matrix / matrix.sum(axis=1)[:, numpy.newaxis]

def normalize_vector(vector):
    denominator = numpy.dot(vector, vector)

    if denominator == 0:
        return numpy.zeros(vector.shape[0])

    return vector / numpy.dot(vector, vector)

def project_vector_onto_vector(project_me, onto):
    numerator = numpy.dot(project_me, onto)
    denominator = numpy.dot(onto, onto)

    if denominator == 0:
        return numpy.zeros(project_me.shape[0])

    return onto * (numerator / denominator)

def get_candidate_anchor_words(vocab_and_wordcounts, doc_threshold):
    # Go through the docs and make sure the words shows up in at least
    #   doc_threshold documents
    candidates = []
    for word, token in enumerate(vocab_and_wordcounts['vocab']):
        docs_found_in = 0
        for document in vocab_and_wordcounts['wordcounts']:
            if document['by_word'].get(word, False):
                docs_found_in += 1

            if (docs_found_in) >= doc_threshold:
                candidates.append(word)
                break

    return candidates

def select_anchors(vocab_and_wordcounts, q, q_norm, anchor_count, projection_dimensions, random):
    """ Selects the anchors from a normalized Q matrix.

    anchor_count is the number of anchors we want to find.

    projection_dimensions is the number of dimensions to project q_norm into.
      This is a tunable hyper parameter.

    random is an instance of numpy.random.RandomState

    The Arora et al. paper https://arxiv.org/abs/1212.4777 describes an anchor
      selection algorithm that involves finding the furthest point from a span.
      I'm not linear algebra expert, but from what I can gather, determining
      such a distance requires orthogonal vectors. Our q_norm rows aren't
      necessarily orthogonal.

    But take heart!  There's a process by which we can take linearly-independent
      vectors and orthogonalize them!
      https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
      So, we'll implment that process.
    """

    candidates = get_candidate_anchor_words(vocab_and_wordcounts, 50)
    print('candidates', candidates)
   
    projected_q_norm = down_project(q_norm, projection_dimensions, random)

    # TODO: consider only considering words that appear in a certain number of
    #   docs
    anchors = []
    max_length = -1.0
    max_index = -1

    # Finds the one closest to the origin
    for i in candidates:
        length = numpy.dot(projected_q_norm[i], projected_q_norm[i])

        if (length > max_length):
            max_length = length
            max_index = i

    #print('first anchor', max_index, projected_q_norm[max_index])
    # First anchor needs to become the origin so that the dot product tells us
    #   distance from span as we start doing the transforms
    for i in range(projected_q_norm.shape[0]):
        projected_q_norm[i] = projected_q_norm[i] - projected_q_norm[max_index]
        #print('centering ', i, ':', projected_q_norm[i])

    anchors.append(max_index)
    projection_basis = normalize_vector(projected_q_norm[max_index])

    # Does the numerically stable Gram-Schmidt described in the Wikipedia
    #   article.  We keep updating all the Qs
    for anchor_index in range(1, anchor_count):
        max_length = -1.0
        max_index = -1

        for i in candidates:
            if i not in anchors:
                projected = project_vector_onto_vector(
                        projected_q_norm[i],
                        projection_basis
                    )

                projected_q_norm[i] = projected_q_norm[i] - projected

                length = numpy.dot(projected_q_norm[i], projected_q_norm[i])

                if length > max_length:
                    max_length = length
                    max_index = i

        anchors.append(max_index)
        projection_basis = normalize_vector(projected_q_norm[max_index])

    print('anchors', anchors)

    return q[anchors, :], anchors

def get_dem_topics(q, q_norm, anchor_words, epsilon=2e-7):
    vocab_size = q.shape[0]
    topic_count = len(anchor_words)
    A = numpy.zeros((vocab_size, topic_count))

    normalization_constants = numpy.diag(q.sum(axis=1))

    for word in range(vocab_size):
        if numpy.isnan(normalization_constants[word, word]):
            normalization_constants[word, word] = 1e-16

    normalized_anchors = normalize_rows(anchor_words)
    normalized_anchors_square = numpy.dot(
            normalized_anchors,
            normalized_anchors.transpose()
        )

    for word in range(vocab_size):
        a_row = exponentiated_gradient(
                q[word, :],
                normalized_anchors,
                normalized_anchors_square,
                epsilon,
            )
        
        if numpy.isnan(a_row).any():
            a_row = numpy.ones(topic_count) / topic_count
        
        A[word, :] = a_row

    A = numpy.matrix(normalization_constants) * numpy.matrix(A)
    for k in range(topic_count):
        A[:, k] = A[:, k] / A[:, k].sum()

    return numpy.array(A)
    
_C1 = 1e-4
_C2 = .75

def exponentiated_gradient(Y, X, XX, epsilon):
    """Solves an exponentied gradient problem with L2 divergence
   
       This function comes from https://github.com/jlund3/ankura/blob/master/ankura/topic.py.

       I renamed many things to make it less generic and more pedagogic.
    """
    XY = numpy.dot(X, Y)
    YY = float(numpy.dot(Y, Y))

    alpha = numpy.ones(X.shape[0]) / X.shape[0]
    old_alpha = numpy.copy(alpha)
    log_alpha = numpy.log(alpha)
    old_log_alpha = numpy.copy(log_alpha)

    AXX = numpy.dot(alpha, XX)
    AXY = float(numpy.dot(alpha, XY))
    AXXA = float(numpy.dot(AXX, alpha.transpose()))

    grad = 2 * (AXX - XY)
    old_grad = numpy.copy(grad)

    new_obj = AXXA - 2 * AXY + YY

    # Initialize book keeping
    stepsize = 1
    decreased = False
    convergence = float('inf')

    while convergence >= epsilon:
        old_obj = new_obj
        old_alpha = numpy.copy(alpha)
        old_log_alpha = numpy.copy(log_alpha)
        if new_obj == 0 or stepsize == 0:
            break

        # Add the gradient and renormalize in logspace, then exponentiate
        log_alpha -= stepsize * grad
        log_alpha -= logsum_exp(log_alpha)
        alpha = numpy.exp(log_alpha)

        # Precompute quantities needed for adaptive stepsize
        AXX = numpy.dot(alpha, XX)
        AXY = float(numpy.dot(alpha, XY))
        AXXA = float(numpy.dot(AXX, alpha.transpose()))

        # See if stepsize should decrease
        old_obj, new_obj = new_obj, AXXA - 2 * AXY + YY
        offset = _C1 * stepsize * numpy.dot(grad, alpha - old_alpha)
        new_obj_threshold = old_obj + offset
        if new_obj >= new_obj_threshold:
            stepsize /= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            new_obj = old_obj
            decreased = True
            continue

        # compute the new gradient
        old_grad, grad = grad, 2 * (AXX - XY)

        # See if stepsize should increase
        if numpy.dot(grad, alpha - old_alpha) < _C2 * numpy.dot(old_grad, alpha - old_alpha) and not decreased:
            stepsize *= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            new_obj = old_obj
            continue

        # Update book keeping
        decreased = False
        convergence = numpy.dot(alpha, grad - grad.min())

    return alpha 

def logsum_exp(y):
    """Computes the sum of y in log space"""
    ymax = y.max()
    return ymax + numpy.log((numpy.exp(y - ymax)).sum())

def get_topic_indices(topics, count):
    topics_with_indices = []
    for i in range(topics.shape[1]):
        topics_with_indices.append(
		[ word for word in numpy.argsort(topics[:, i])[-count:][::-1] ]
	    )

    return topics_with_indices

def topic_indices_to_words(topics, vocab):
    return [ [ vocab[i] for i in topic ] for topic in topics ]

def load_the_data(dataset_name, inclusion_threshold=1):
    print('load the data with threshold:', inclusion_threshold)
    dataset_pickle_name = dataset_name + '.pkl'

    print('checking for pickle file', dataset_pickle_name)

    if os.path.isfile(dataset_pickle_name):
        print('we have it, just load that')

        with open(dataset_pickle_name, 'rb') as f:
            return pickle.load(f)
    else:
        print('we do not have it :(')

        print('load raw data')
        raw_data = tokenize_dataset(dataset_name)

        vocab_and_wordcounts = build_vocab_and_wordcounts(raw_data, inclusion_threshold=inclusion_threshold)

        print('save', dataset_pickle_name)
        with open(dataset_pickle_name, 'wb') as f:
            pickle.dump(vocab_and_wordcounts, f, pickle.HIGHEST_PROTOCOL)

        return vocab_and_wordcounts

def process_dataset(dataset_name, random, inclusion_threshold=1):
    print('inclusion threshold:', inclusion_threshold)
    vocab_and_wordcounts = load_the_data(dataset_name, inclusion_threshold=inclusion_threshold)

    print('vocab_size', len(vocab_and_wordcounts['vocab']))

    print('build q')
    q = build_q(vocab_and_wordcounts)

    print('normalize q')
    q_norm = normalize_rows(q)

    num_topics = 20
    projection_dimensions = 1000

    anchors, anchor_indices = select_anchors(vocab_and_wordcounts, q, q_norm, num_topics, projection_dimensions, random)

    print('get the topics')
    topics = get_dem_topics(q, q_norm, anchors)

    print('first topic length', len(topics[0]))

    topic_indices = get_topic_indices(topics, 10)
    topic_words = topic_indices_to_words(
            topic_indices,
            vocab_and_wordcounts['vocab'],
        )

    coherences = calculate_coherences(
            vocab_and_wordcounts['wordcounts'],
            topic_indices,
        )

    print('coherence sum', sum(coherences))

    for i in range(len(topic_indices)):
        print('--------------')
        print(topic_words[i])
        print(coherences[i])

def calculate_coherences(wordcounts, topic_indices):
    return [ calculate_coherence(wordcounts, topic) for topic in topic_indices ]

def calculate_coherence(wordcounts, topic_indices):
    coherence = 0
    print('calculate_coherence', topic_indices)

    for i in topic_indices:
        for j in topic_indices:
            if i == j:
                continue

            def doc_contains_word(memo, word):
                if word['by_word'].get(j, 0) > 0:
                    return memo + 1
                else:
                    return memo

            def doc_contains_both(memo, word):
                if word['by_word'].get(j, 0) > 0 and word['by_word'].get(i, 0) > 0:
                    return memo + 1
                else:
                    return memo

            docs_with_j = reduce(doc_contains_word, wordcounts, 0)
            docs_with_both = reduce(doc_contains_both, wordcounts, 0)

            coherence += math.log((docs_with_both + .1) / docs_with_j)

    return coherence

