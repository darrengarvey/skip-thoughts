#!/usr/bin/env python

import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
from vocab import Vocab


# Whether to use tf.contrib.layers or not.
tf_layers = False
# Change this
base_dir = 'data/skip-thought-small-60'

def get_input(pattern, features, shuffle=False):
    examples = tf.contrib.learn.io.read_keyed_batch_features(
        pattern,
        batch_size=1,
        features=features,
        randomize_input=shuffle,
        reader=tf.TFRecordReader)
    return examples


vocab = Vocab(base_dir+'/vocab.txt')
vocab_size = len(vocab)

if tf_layers:
    encode = layers.sparse_column_with_integerized_feature(
        'encode', bucket_size=vocab_size)
    decode_pre = layers.sparse_column_with_integerized_feature(
        'decode_pre', bucket_size=vocab_size)
    decode_post = layers.sparse_column_with_integerized_feature(
        'decode_post', bucket_size=vocab_size)
    features = {
        'encode': encode,
        'encode_emb': layers.embedding_column(encode, dimension=100),
        'decode_pre': layers.embedding_column(decode_pre, dimension=100),
        'decode_post': layers.embedding_column(decode_post, dimension=100),
    }
    features = tf.contrib.layers.create_feature_spec_for_parsing(features)
else:
    # This little dict seems equivalent to the waay more verbose
    # tf.contrib.layers approach. But apparently the latter helps,
    # especially when it comes to tf serving. Still to see the benefit...
    features = {
        "encode": tf.VarLenFeature(dtype=tf.int64),
        "decode_pre": tf.VarLenFeature(dtype=tf.int64),
        "decode_post": tf.VarLenFeature(dtype=tf.int64),
    }
i = get_input(base_dir+'/validation-00000-of-00001', features)

def decode_sentence(tensor, vocab):
    """Turn a list of ids into a string of words"""
    return ' '.join(vocab.decode(tensor.values))

def get_sequence_lengths(tensor):
    # There's got to be a cleaner way to get the sequence lengths...
    _, _, counts = tf.unique_with_counts(tensor.indices[:,0])
    return counts.eval()

def print_decode(examples, name, vocab):
    print ('{} (len={}): {}'.format(
        name, get_sequence_lengths(examples[name]),
        decode_sentence(examples[name], vocab)))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    with open('sentences.txt', 'w') as f:
        while True:
            files, examples = sess.run(i)
            f.write('{}\n'.format(decode_sentence(examples['encode'], vocab)))
