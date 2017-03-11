from functools import reduce 
import scipy.sparse

from get_files_in_dataset import get_files_in_dataset
from tokenize_file import tokenize_file

# This creates a vocab and a list of document word counts from a path to a dir.
def build_dataset(dataset):
    vocab = {}

    docs = []

    dataset_files = get_files_in_dataset(dataset)

    # * `vocab` maps from the actual tokens in the docs to an index
    # * `docs` is an array of dictionaries whose keys are numbers. The numbers
    #     correspond to what you'd get if you asked `vocab` for the actual
    #     token.
    initial = { "vocab": {}, "doc_word_counts": [] }

    # reduction function
    def reduce_file(memo, path):
        print("Reducing:", path)

        file_content = tokenize_file(path)
        doc_word_count = {}

        # now I have all the words.  I need to apply them
        #  To the current doc and the vocab
        for word in file_content:
            # Add to the vocab if we don't already have it
            if word not in memo["vocab"]:
                memo["vocab"][word] = len(memo["vocab"])

            # add it to the current doc's count
            word_id = memo["vocab"][word]
            doc_word_count[word_id] = doc_word_count.get(word_id, 0) + 1

        memo["doc_word_counts"].append(doc_word_count)

        return memo

    vocab_and_doc_word_counts = reduce(reduce_file, dataset_files, initial)
    doc_word_counts = vocab_and_doc_word_counts["doc_word_counts"]

    # Turns the vocab into a list of words.  List position of a word
    #   corresponds to the value the word had in the word => number mapping
    #   during the reduction.
    vocab_id_map = vocab_and_doc_word_counts["vocab"]
    sorted_vocab_items = sorted(vocab_id_map.items(), key=lambda v: v[1])
    vocab_list = [v[0] for v in sorted_vocab_items]

    vocab_document_matrix = build_vocab_document_matrix(vocab_list,
                                                        doc_word_counts)

    return { "vocab_document_matrix": vocab_document_matrix.tocsc(),
             "vocab_list": vocab_list }

def build_vocab_document_matrix(vocab_list, doc_word_counts):
    """Given a vocab list and a list of document word count dictionaries,
    this function returns the matrix represenation of those inputs"""

    matrix = scipy.sparse.lil_matrix((len(vocab_list), len(doc_word_counts)),
                                     dtype='uint')

    for doc_id, doc_word_count in enumerate(doc_word_counts):
        for word_id, word_count in doc_word_count.items():
            matrix[word_id, doc_id] = word_count

    return matrix

