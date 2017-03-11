import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

def tokenize_file(path_to_file):
  """Takes a file path and returns a tokenized/lemmatized list"""

  lemmatizer = WordNetLemmatizer()

  file_object = open(path_to_file, 'r')
  contents = file_object.read()
  no_punctuation = "".join(l for l in contents if l not in string.punctuation)
  tokenized = nltk.word_tokenize(no_punctuation)
  lowercased = [w.lower() for w in tokenized]
  lemmatized = [lemmatizer.lemmatize(w) for w in lowercased]
  without_stop_words = [word for word in lemmatized if word not in stopwords.words('english')]

  return without_stop_words
