from functools import reduce 
import numpy
import scipy

def reduce_file_vocab_and_wordcount(memo, document):
    by_word = {}

    for word in document:
        #print('word', word)
        if word not in memo['seen_vocab']:
            memo['seen_vocab'][word] = len(memo['vocab'])
            memo['vocab'].append(word)

        word_id = memo['seen_vocab'][word]
        by_word[word_id] = by_word.get(word_id, 0) + 1

    wordcount = { 'total': len(document), 'by_word': by_word }

    memo['wordcounts'].append(wordcount)

    return memo

def build_vocab_and_wordcounts(documents):
    vocab_and_wordcounts = reduce(
            reduce_file_vocab_and_wordcount,
            documents,
            { 'vocab': [], 'wordcounts': [], 'seen_vocab': {} }
        )

    del vocab_and_wordcounts['seen_vocab']

    return vocab_and_wordcounts

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
    return numpy.array(h_tilde * h_tilde.transpose() - numpy.diag(h_hat))

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

def select_anchors(q, q_norm, anchor_count, projection_dimensions, random):
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
   
    projected_q_norm = down_project(q_norm, projection_dimensions, random)

    # TODO: consider only considering words that appear in a certain number of
    #   docs
    anchors = []
    max_length = -1.0
    max_index = -1

    for i in range(projected_q_norm.shape[0]):
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

        for i in range(1, projected_q_norm.shape[0]):
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

    return q[anchors, :], anchors

def get_dem_topics(q, q_norm, anchor_words, epsilon=2e-7):
    vocab_size = q.shape[0]
    topic_count = len(anchor_words)
    A = numpy.zeros((vocab_size, topic_count))

    normalization_constants = numpy.diag(q.sum(axis=1))

    # Don't let there be zeroes?

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

def process_dataset(raw_data, random):
    print('vocab and wordcounts')
    vocab_and_wordcounts = build_vocab_and_wordcounts(raw_data)

    print('build q')
    q = build_q(vocab_and_wordcounts)
    q_norm = normalize_rows(q)

    num_topics = 2
    projection_dimensions = 4

    anchors = select_anchors(q, q_norm, num_topics, projection_dimensions, random)

    print('get the topics')
    topics = get_dem_topics(q, q_norm, anchors)

    topic_indices = get_topic_indices(topics, 10)
    topic_words = topic_indices_to_words(topics, vocab_and_wordcounts['vocab'])

    print(topic_words)
