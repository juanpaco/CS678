import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os import listdir
from os.path import isfile, join
import pickle
import string

# path should be the name of a directory under "data"
def get_files_in_dataset(dataset_name):
  path = join("data", dataset_name)
  file_paths = [f for f in listdir(path) if isfile(join(path, f))]
  full_file_paths = list(map(lambda f: join(path, f), file_paths))

  return full_file_paths

def tokenize_dataset(dataset_name):
    """Returns an iterator whose items are the tokenized files"""

    raw_data_pickle_file = dataset_name + '.raw.pkl'

    print('checking for raw data:', raw_data_pickle_file)
    if isfile(raw_data_pickle_file):
        print('we have it, just use that')

        with open(raw_data_pickle_file, 'rb') as f:
            return pickle.load(f)
    else:
        print('we do not have it :(.  So load it')

        raw_data = list(map(tokenize_file, get_files_in_dataset(dataset_name)))

        print('save raw', raw_data_pickle_file)
        with open(raw_data_pickle_file, 'wb') as f:
            pickle.dump(raw_data, f, pickle.HIGHEST_PROTOCOL)

        return raw_data


def tokenize_file(path_to_file):
  """Takes a file path and returns a tokenized/lemmatized list"""

  #print('tokenizing', path_to_file)

  lemmatizer = WordNetLemmatizer()

  file_object = open(path_to_file, 'r')
  contents = file_object.read()
  no_punctuation = "".join(l for l in contents if l not in string.punctuation)
  tokenized = nltk.word_tokenize(no_punctuation)
  lowercased = [w.lower() for w in tokenized]
  lemmatized = [lemmatizer.lemmatize(w) for w in lowercased]
  without_stop_words = [word for word in lemmatized if word not in stopwords.words('english')]

  return without_stop_words

