import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.state_saving_rnn_estimator import StateSavingRnnEstimator
import util


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

  def get_predictions(self, features, targets, mode, params):
    """Build and return the model."""
    # decode_pre_emb, decode_pre_mask, decode_pre_targets
    # decode_post_emb, decode_post_mask, decode_post_targets
    encode_input, encode_mask = self._setup_input()
    thought_vector = self._setup_encoder(encode_input, encode_mask)
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
      # When training, feed the thought vector into two separate
      # networks: predicting the previous and next sentences
      decoder_pre = self._setup_decoder(
          'decoder_pre', thought_vector,
          decode_pre_emb, decode_pre_mask,
          reuse=False)
      decoder_post = self._setup_decoder(
          'decoder_post', thought_vector,
          decode_post_emb, decode_post_mask,
          reuse=True)
      return {
          'decoder_pre_logits': decoder_pre[0],
          'decoder_pre_mask': decoder_pre[1],
          'decoder_post_logits': decoder_post[0],
          'decoder_post_mask': decoder_post[1],
      }, {
          'decoder_pre_targets': decoder_pre_targets,
          'decoder_post_targets': decode_post_targets,
      }

  def get_loss(self, predictions, targets, mode, params):
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      decoder_pre_loss = self._decoder_loss(
          'decoder_pre', predictions['decoder_pre_logits'],
          predictions['decoder_pre_mask'], targets['decoder_pre_targets'])
      decoder_post_loss = self._decoder_loss(
          'decoder_post', predictions['decoder_post_logits'],
          predictions['decoder_post_mask'], targets['decoder_post_targets'])
      total_loss = decoder_pre_loss + decoder_post_loss
      tf.summary.scalar('losses/total', total_loss)
      return total_loss
 
  def _setup_decoder(self, name, initial_state, embeddings, mask, reuse):
    num_units = self.params['encoder_dim']
    cell = self._get_cell(num_units)
    with tf.variable_scope(name) as scope:
      decoder_input = tf.pad(
          embeddings[:, :-1. :], [[0, 0], [1, 0], [0, 0]], name='input')
      length = tf.reduce_sum(mask, 1, name='length')
      decoder_output, _ = tf.nn.dynamic_rnn(
          cell=cell,
          inputs=decoder_input,
          sequence_length=length,
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
      return logits, mask

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

  def _setup_encoder(self, inputs, encode_mask):
    with tf.variable_scope('encoder') as scope:
      length = tf.to_int32(tf.reduce_sum(encode_mask, 1), name='length')
      if self.params['bidirectional_decoder']:
        num_units = self.params['encoder_dim'] // 2
        cell_fw = self._get_cell(num_units)
        cell_bw = self._get_cell(num_units)
        _, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope)
        thought_vectors = tf.concat(states, 1, name='thought_vectors')
      else:
        cell = self._get_cell(self.params['encoder_dim'])
        _, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope)
        thought_vectors = tf.identity(state, name='thought_vectors')
    return thought_vectors

  def _get_input(self, filename):
    pass

  def get_training_input(self):
    return self._get_input('train.tfrecords')

  def get_eval_input(self):
    return self._get_input('validation.tfrecords')

  def get_predictions(self, features, targets, mode, params):
    pass

  def get_loss(self, predictions, targets, mode, params):
    pass
