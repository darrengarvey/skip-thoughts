import os
from nltk.tokenize import word_tokenize


class Vocab(object):
  def __init__(self, vocab_file, unk_token='<unk>', eos_token='<eos>'):
    with open(vocab_file, 'r') as f:
      self._id_to_word = [l.strip() for l in f.readlines()]
      self._word_to_id = {
          word: id
          for id,word in enumerate(self._id_to_word)
      }
      self._unk = self._word_to_id[unk_token]
      self._eos = self._word_to_id[eos_token]

  def __len__(self):
    return len(self._id_to_word)

  def __iter__(self):
    return iter(self._id_to_word)

  def __getitem__(self, key):
    return self._id_to_word[key]

  def word(self, id):
    """Map an id to a word."""
    return self._id_to_word[id]

  def id(self, word):
    """Map a word to an id, or the <unk> token."""
    return self._word_to_id.get(word, self._unk)

  def encode(self, sentence, add_eos=False):
    """Encode a string sentence into a vector of ids."""
    words = word_tokenize(sentence)
    ids = map(self.id, words)
    if add_eos:
      ids.append(self._eos)
    return ids

  def decode(self, ids):
    """Decode a vector of ids into a vector of words."""
    return map(self.word, ids)
