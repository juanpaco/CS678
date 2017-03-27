from anchor_words import (
        build_vocab_and_wordcounts,
        build_q,
        reduce_file_vocab_and_wordcount,
    )
import pytest

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
