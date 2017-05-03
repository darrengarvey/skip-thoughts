#!/usr/bin/env python

"""
Save word embeddings for all words in the vocab into an annoy DB, so nearest
neighbours and embedding maths can be done with it.
"""

import argparse
import os
import sys
import tensorflow as tf
from annoy import AnnoyIndex
from tqdm import tqdm
# Local imports
from vocab import Vocab
from interactive import get_input


def parse_args(args):
  parser = argparse.ArgumentParser(description=__doc__)
  # Add additional command line stuff here...
  parser.add_argument('--logdir', required=True,
                      help='Location to store checkpoint / log data')
  parser.add_argument('--batch-size', default=128, type=int,
                      help='Fetch embeddings in batches (default=%(default)s)')
  parser.add_argument('--num-trees', default=10, type=int,
                      help='Number of trees in the index. Higher means more '
                           'accuracy but also more memory (default=%(default)s)')
  parser.add_argument('--embedding-var_name', default='word_embedding:0',
                      help='Word embedding variable name (default=%(default)s)')
  parser.add_argument('--embedding-dim', default=620, type=int,
                      help='Word embedding dimension (default=%(default)s)')
  parser.add_argument('--embedding-filename', default='encoded-words.ann',
                      help='Annoy index containing word embeddings (default=%(default)s)')
  parser.add_argument('-k', '--search-k', default=-1, type=int,
                      help='Number of bins to search. Larger k is slower but '
                           'more precise. (default=%(default)s)')
  parser.add_argument('--vocab', default='vocab.txt', type=str,
                      help='Path to the vocab file used to encode the input '
                           '(default=<logdir>/%(default)s)')
  return parser.parse_args(args)


class WordEmbeddings(object):
  def __init__(self, args):
    self.__dict__.update(vars(parse_args(args)))
    if '/' in self.vocab:
      # Unfortunately this won't work if the user just passes in a file in
      # the current directory, but in most cases this logic means the user
      # won't have to pass in vocab most of the time.
      vocab_dir = self.vocab
    else:
      vocab_dir = os.path.join(self.logdir, self.vocab)
    self.vocab = Vocab(vocab_dir)
    self.embedding_file = os.path.join(self.logdir, self.embedding_filename)
    self._restore_checkpoint()
    self.index = AnnoyIndex(self.embedding_dim)
    if not os.path.exists(self.embedding_file):
      print ('Embeddings not already indexed. Doing this now (note this '
             'will take a few minutes)')
      self._save_word_embeddings()
    else:
      self._load_word_embeddings()

  def nearest_words(self, line, n=10):
    """Find the nearest n words to a word or an expression like
       "kind - man + woman". Only very basic maths is allowed and
       no grouping or precedence is handled properly."""
    words = line.split(' ')
    embedding = None
    print ('Found {} tokens'.format(len(words)))
    if 1 == len(words):
      word_id = self.vocab.id(words[0])
      word_emb = self._lookup(tf.constant(word_id))
      embedding = self.session.run(word_emb)
    else:
      emb = None
      op = None
      for word in words:
        if word == '-':
          op = tf.subtract
        elif word == '+':
          op = tf.add
        elif word == '/':
          op = tf.divide
        elif word == '*':
          op = tf.multiply
        else:
          word_id = self.vocab.id(word)
          word_emb = self._lookup(tf.constant(word_id))
          if op:
            emb = op(emb, word_emb)
            op = None
          else:
            emb = word_emb
      embedding = self.session.run(emb)
    #print ('embedding', embedding.shape, embedding)
    nn, distances = self.index.get_nns_by_vector(
        embedding, n, search_k=self.search_k, include_distances=True)
    return [(self.vocab.word(n), dist) for n, dist in zip(nn, distances)]

  def _lookup(self, tensor):
    return tf.nn.embedding_lookup(self.embedding_var, tensor)

  def _load_word_embeddings(self):
    self.index.load(self.embedding_file)

  def _restore_checkpoint(self):
    checkpoint = tf.train.latest_checkpoint(self.logdir)
    print ('Loading embedding variable from checkpoint: {}'.format(checkpoint))
    self.session = tf.Session()
    self.embedding_var = tf.contrib.framework.load_variable(
        checkpoint, self.embedding_var_name)

  def _save_word_embeddings(self):
    vocab_len = len(self.vocab)
    for i in tqdm(range(0, vocab_len, self.batch_size), desc='batches'):
      max_i = min(vocab_len, i + self.batch_size)
      r = tf.range(i, max_i)
      word_emb = self._lookup(r)
      embeddings = self.session.run(word_emb)
      #print ('embeddings', embeddings.shape, embeddings)
      for j, embedding in tqdm(enumerate(embeddings), desc='words'):
        #print ('embedding', i + j, embedding.shape, embedding)
        self.index.add_item(i + j, embedding)
  
    self.index.build(self.num_trees)
    self.index.save(self.embedding_file)

def main(args):
    emb = WordEmbeddings(args)
    print ('Loaded embeddings. Enter a word to find nearest neighbours')
    for line in get_input():
      print (line)
      line = line.lower().strip()
      if len(line):
        matches = emb.nearest_words(line)
        for word, dist in matches:
          print ('Match (d={:.4f}): {}'.format(dist, word))



if __name__ == '__main__':
  main(sys.argv[1:])
