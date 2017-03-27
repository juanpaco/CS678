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
        #print(doc_id, ':', wordcount)
        doc_length = 0
        normalization_factor = wordcount['total'] * (wordcount['total'] - 1)

        for word_id, count in wordcount['by_word'].items():
            doc_length += count

            h_tilde[word_id, doc_id] = count / numpy.sqrt(normalization_factor)
            h_hat[word_id] += count / normalization_factor

            #print(word_id, ':', count)


    #print('vocab', vocab)
    #print('h_tilde', h_tilde)
    #print('h_hat', h_hat)
    return h_tilde * h_tilde.transpose() - numpy.diag(h_hat)
