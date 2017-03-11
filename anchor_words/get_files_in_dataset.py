from os import listdir
from os.path import isfile, join

# path should be the name of a directory under "data"
def get_files_in_dataset(dataset_name):
  path = join("data", dataset_name)
  file_paths = [f for f in listdir(path) if isfile(join(path, f))]
  full_file_paths = list(map(lambda f: join(path, f), file_paths))

  return full_file_paths
