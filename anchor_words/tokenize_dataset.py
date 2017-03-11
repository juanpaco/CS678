from os.path import join

import get_files_in_directory

def tokenize_dataset(dataset):
    dataset_path = "data/" + dataset
    files = get_files_in_directory.get_files_in_directory(dataset_path)
    full_file_paths = list(map(lambda f: join(dataset_path, f), files))

    print("tokenize dataset", full_file_paths)
