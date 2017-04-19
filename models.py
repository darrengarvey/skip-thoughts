import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.learn.python.learn.estimators.state_saving_rnn_estimator import StateSavingRnnEstimator
import util
from gru_cell import LayerNormGRUCell


def random_orthonormal_initializer(shape, dtype=tf.float32,
                                   partition_info=None):
  """Variable initializer that produces a random orthonormal matrix."""
  if len(shape) != 2 or shape[0] != shape[1]:
    raise ValueError("Expecting square shape, got %s" % shape)
  _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
  return u


class SkipThoughtsModel(util.Model):
  def __init__(self, args):
    super(SkipThoughtsModel, self).__init__(args)
    self.uniform_initializer = tf.random_uniform_initializer(
        minval=-self.uniform_init_scale,
        maxval=self.uniform_init_scale)

    with open(args.vocab, 'r') as f:
        self.vocab = [l.strip() for l in f]
    # Set up input parsing stuff via tf.contrib.learn...
    self.vocab_size = len(self.vocab)
    encode = layers.sparse_column_with_integerized_feature(
        'encode', bucket_size=self.vocab_size)
    decode_pre = layers.sparse_column_with_integerized_feature(
        'decode_pre', bucket_size=self.vocab_size)
    decode_post = layers.sparse_column_with_integerized_feature(
        'decode_post', bucket_size=self.vocab_size)
    self.features = {
        'encode': layers.embedding_column(encode, dimension=100),
        'decode_pre': layers.embedding_column(decode_pre, dimension=100),
        'decode_post': layers.embedding_column(decode_post, dimension=100),
    }
    self.feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(self.features),
    # ... or do it the easy way:
    self.features = {
        "encode": tf.VarLenFeature(dtype=tf.int64),
        "decode_pre": tf.VarLenFeature(dtype=tf.int64),
        "decode_post": tf.VarLenFeature(dtype=tf.int64),
    }
    self.feature_spec = self.features

  def _get_input(self, pattern):
    """Read, parse and batch tf.Examples in multiple threads and push them onto
       a queue."""
    # Ignore the returned file names for now.
    _, examples = tf.contrib.learn.io.read_keyed_batch_features(
        pattern,
        batch_size=self.batch_size,
        features=self.feature_spec,
        reader=tf.TFRecordReader)

    features = {
        'encode': examples['encode'],
    }
    targets = {
        'decode_pre': examples['decode_pre'],
        'decode_post': examples['decode_post'],
    }
    return features, targets

  def get_training_input(self):
    return self._get_input(self.input_pattern)

  def get_eval_input(self):
    return self._get_input('validation.tfrecords')

  def _get_sequence_lengths(self, tensor):
    _, _, counts = tf.unique_with_counts(tensor.indices[:,0])
    return counts

  def get_predictions(self, features, targets, mode, params):
    """Build and return the model."""
    # Share the word embedding between encoder and decoder.
    word_emb = tf.get_variable(
        'word_embedding',
        shape=[self.vocab_size, params['embedding_dim']],
        initializer=self.uniform_initializer)
    encode_input = tf.nn.embedding_lookup_sparse(
        word_emb, features['encode'], None)
    encode_lengths = self._get_sequence_lengths(features['encode'])
    # Now encode a thought vector and feed it into the two decoders.
    thought_vector = self._setup_encoder(encode_input, encode_lengths)

    if mode != tf.contrib.learn.ModeKeys.TRAIN:
      # When training, feed the thought vector into two separate
      # networks: predicting the previous and next sentences
      decode_pre_input = tf.nn.embedding_lookup(word_emb, targets['decode_pre'])
      decode_pre_lengths = self._get_sequence_lengths(targets['decode_pre'])
      decode_pre = self._setup_decoder(
          'decode_pre', thought_vector,
          decode_pre_input, decode_pre_lengths,
          reuse=False)
      decode_post_input = tf.nn.embedding_lookup(word_emb, targets['decode_post'])
      decode_post_lengths = self._get_sequence_lengths(targets['decode_post'])
      decode_post = self._setup_decoder(
          'decode_post', thought_vector,
          decode_post_input, decode_post_lengths,
          reuse=True)
      return {
          'decode_pre_logits': decode_pre,
          'decode_pre_lengths': decode_pre_lengths,
          'decode_post_logits': decode_post,
          'decode_post_lengths': decode_post_lengths,
      }, {
          'decode_pre_targets': targets['decode_pre'],
          'decode_post_targets': targets['decode_post'],
      }

  def get_loss(self, predictions, targets, mode, params):
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      decode_pre_loss = self._decoder_loss(
          'decode_pre', predictions['decode_pre_logits'],
          predictions['decode_pre_mask'], targets['decode_pre_targets'])
      decode_post_loss = self._decoder_loss(
          'decode_post', predictions['decode_post_logits'],
          predictions['decode_post_mask'], targets['decode_post_targets'])
      total_loss = decode_pre_loss + decode_post_loss
      tf.summary.scalar('losses/total', total_loss)
      return total_loss
    else:
      raise Exception("Wrong mode {}".format(mode))
 
  def _setup_decoder(self, name, initial_state, embeddings, sequence_lengths, reuse):
    num_units = self.params['encoder_dim']
    cell = self._get_cell(num_units)
    with tf.variable_scope(name) as scope:
      decoder_input = tf.pad(
          embeddings[:, :-1. :], [[0, 0], [1, 0], [0, 0]], name='input')
      decoder_output, _ = tf.nn.dynamic_rnn(
          cell=cell,
          inputs=decoder_input,
          sequence_length=sequence_lengths,
          initial_state=initial_state,
          scope=scope)

    # Stack batch vertically.
    decoder_output = tf.reshape(decoder_output, [-1, num_units])

    # Logits.
    with tf.variable_scope('logits', reuse=reuse) as scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=decoder_output,
          num_outputs=self.params['vocab_size'],
          activation_fn=None,
          weights_initializer=self.uniform_initializer,
          scope=scope)
      return logits

  def _decoder_loss(self, name, logits, mask, targets):
    targets = tf.reshape(targets, [-1])
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits)
    weights = tf.to_float(tf.reshape(mask, [-1]))
    batch_loss = tf.reduce_sum(losses * weights)
    tf.summary.scalar('losses/' + name, batch_loss)
    return batch_loss

  def _get_cell(self, num_units):
    cell = LayerNormGRUCell(
        num_units,
        w_initializer=self.uniform_initializer,
        u_initializer=random_orthonormal_initializer,
        b_initializer=tf.constant_initializer(0.0))
    # TODO: Add AttentionWrapper, DropoutWrapper, etc.
    return cell

  def _setup_encoder(self, inputs, sequence_lengths):
    with tf.variable_scope('encoder') as scope:
      if self.params['bidirectional']:
        print('bidirectional')
        num_units = self.params['embedding_dim'] // 2
        cell_fw = self._get_cell(num_units)
        cell_bw = self._get_cell(num_units)
        _, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=sequence_lengths,
            dtype=tf.float32,
            scope=scope)
        thought_vectors = tf.concat(states, 1, name='thought_vectors')
      else:
        cell = self._get_cell(self.params['embedding_dim'])
        _, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            sequence_length=sequence_lengths,
            dtype=tf.float32,
            scope=scope)
        thought_vectors = tf.identity(state, name='thought_vectors')
    return thought_vectors
