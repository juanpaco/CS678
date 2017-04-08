import numpy
import pytest

from anchor_words import (
        build_vocab_and_wordcounts,
        build_q,
        down_project,
        normalize_rows,
        normalize_vector,
        project_vector_onto_vector,
        reduce_file_vocab_and_wordcount,
        select_anchors,
    )

from read_dataset import (tokenize_dataset)

def test_build_vocab_and_wordcounts():
    documents = [
            [ 'this', 'is', 'data', 'this' ],
            [ 'this', 'is', 'another' ],
        ]

    vocab_and_wordcounts = build_vocab_and_wordcounts(documents)

    assert len(vocab_and_wordcounts['vocab']) == 4

    # Check the first doc for its count of 'this'
    assert vocab_and_wordcounts['wordcounts'][0]['by_word'][0] == 2
    assert vocab_and_wordcounts['wordcounts'][0]['total'] == 4

def test_build_q():
    documents = [
            [ 'this', 'is', 'data', 'this' ],
            [ 'this', 'is', 'another' ],
        ]

    vocab_and_wordcounts = build_vocab_and_wordcounts(documents)
    q = build_q(vocab_and_wordcounts)
    vocab_length = len(vocab_and_wordcounts['vocab'])

    # q = h_tilde * h_tilde.transpose - h_hat
    # h_tilde is (vocab_length, num_docs), so it times its transpose is
    #   (vocab_length, vocab_length). h_hat is the diag of a vector with length
    #   vocab_length, so we end up with a (vocab_length, vocab_length) +
    #   (vocab_length, vocab_length).
    assert q.shape == (vocab_length, vocab_length)

    # 'another' and 'data' never show up in the same doc
    assert q[2,3] == 0
    assert q[3,2] == 0

def test_another_build_q():
    """ This tests against a known dataset """

    documents = [
            [ 'dog', 'dog' ],
            [ 'dog', 'cat' ],
        ]

    vocab_and_wordcounts = build_vocab_and_wordcounts(documents)
    q = build_q(vocab_and_wordcounts)
    vocab_length = len(vocab_and_wordcounts['vocab'])

    # q = h_tilde * h_tilde.transpose - h_hat
    # h_tilde is (vocab_length, num_docs), so it times its transpose is
    #   (vocab_length, vocab_length). h_hat is the diag of a vector with length
    #   vocab_length, so we end up with a (vocab_length, vocab_length) +
    #   (vocab_length, vocab_length).
    assert q.shape == (vocab_length, vocab_length)

    assert q[0,0] == pytest.approx(1, abs=.001)

def test_reduce_file_vocab_and_wordcount():
    vocab_and_wordcounts = { 'vocab': [], 'wordcounts': [], 'seen_vocab': {} }

    vocab_and_wordcounts = reduce_file_vocab_and_wordcount(
            vocab_and_wordcounts,
            [ 'this', 'is', 'data', 'this' ],
        )

    assert len(vocab_and_wordcounts['vocab']) == 3
    assert vocab_and_wordcounts['seen_vocab']['data'] == 2
    assert vocab_and_wordcounts['wordcounts'][0]['by_word'][0] == 2
    assert vocab_and_wordcounts['wordcounts'][0]['total'] == 4

    vocab_and_wordcounts = reduce_file_vocab_and_wordcount(
            vocab_and_wordcounts,
            [ 'this', 'is', 'another' ],
        )

    assert len(vocab_and_wordcounts['vocab']) == 4
    assert vocab_and_wordcounts['seen_vocab']['another'] == 3
    assert vocab_and_wordcounts['wordcounts'][1]['by_word'][0] == 1
    assert vocab_and_wordcounts['wordcounts'][1]['by_word'][3] == 1
    assert vocab_and_wordcounts['wordcounts'][1]['total'] == 3

def test_pick_anchor_words():
    print('hello')

def test_down_project():
    random = numpy.random.RandomState(1)

    matrix = numpy.ones((4, 4))

    projected = down_project(matrix, 2, random)

    assert projected.item((0,0)) == 0
    assert projected.item((0,1)) == pytest.approx(1.732, abs=.001)

def test_normalize_rows():
    matrix = numpy.matrix('1 2 3; 3 8 7', dtype=float)

    normalized = normalize_rows(matrix)

    assert normalized.item((0,2)) == pytest.approx(.5, abs=.01)

def test_normalize_vector():
    vector = numpy.array([ 3, 4 ])

    normie_vector = normalize_vector(vector)

    assert normie_vector[0] == pytest.approx(3.0/25.0, abs=.01)

def test_normalize_zero_vector():
    vector = numpy.array([ 0, 0 ])

    normie_vector = normalize_vector(vector)

    assert normie_vector[0] == 0

def test_project_vector_onto_vector():
    project_me = numpy.array([ 1, 1 ])
    onto = numpy.array([ 1, 0 ])

    projected = project_vector_onto_vector(project_me, onto)

    assert projected[0] == 1
    assert projected[1] == 0

def test_project_vector_onto_zero_matrix():
    project_me = numpy.array([ 1, 1 ])
    onto = numpy.array([ 0, 0 ])

    projected = project_vector_onto_vector(project_me, onto)

    assert projected[0] == 0
    assert projected[1] == 0

def test_select_anchors():
    random = numpy.random.RandomState(1)
    raw_data = tokenize_dataset('test')
    vocab_and_wordcounts = build_vocab_and_wordcounts(raw_data)
    q = build_q(vocab_and_wordcounts)
    q_norm = normalize_rows(q)

    num_topics = 2
    projection_dimensions = 4

    anchors = select_anchors(q_norm, num_topics, projection_dimensions, random)

    assert len(anchors) == num_topics
    assert vocab_and_wordcounts['vocab'][anchors[0]] == 'file'
    assert vocab_and_wordcounts['vocab'][anchors[1]] == 'dog'
