from read_dataset import (
        get_files_in_dataset,
        tokenize_dataset,
        tokenize_file,
    )

def test_tokenize_dataet():
    tokenized = tokenize_dataset('test')

    first = next(tokenized)

    assert first[0] == 'also'

def test_tokenize_file():
    paths = sorted(get_files_in_dataset('test'))

    tokenized = tokenize_file(paths[0])

    assert tokenized[0] == 'file'

def test_get_files_in_dataset():
    paths = sorted(get_files_in_dataset('test'))

    assert paths[0] == 'data/test/file-1.txt'
    assert paths[1] == 'data/test/file-2.txt'
