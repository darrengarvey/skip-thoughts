import tensorflow as tf
import tensorflow.contrib.layers as layers
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
        'encode': tf.VarLenFeature(dtype=tf.int64),
        'decode_pre': tf.VarLenFeature(dtype=tf.int64),
        'decode_post': tf.VarLenFeature(dtype=tf.int64),
    }
    self.feature_spec = self.features

  def _sparse_to_batch(self, sparse):
    print ('shapes', sparse.dense_shape)
    ids = tf.sparse_tensor_to_dense(sparse)
    mask = tf.sparse_to_dense(sparse.indices, sparse.dense_shape,
                              tf.ones_like(sparse.values, dtype=tf.int32))
    return ids, mask

  def _get_input(self, pattern, name=None, num_epochs=None, shuffle=True):
    """Read, parse and batch tf.Examples in multiple threads and push them onto
       a queue."""
    # Ignore the returned file names for now.
    examples = tf.contrib.learn.io.read_batch_record_features(
        file_pattern=pattern,
        batch_size=self.batch_size,
        features=self.feature_spec,
        reader_num_threads=2, # one thread can't keep up with >1 GPUs
        num_epochs=num_epochs,
        randomize_input=shuffle,
        name=name)

    features = {
        'encode': tf.sparse_tensor_to_dense(examples['encode']),
    }
    ids1, mask1 = self._sparse_to_batch(examples['decode_pre'])
    ids2, mask2 = self._sparse_to_batch(examples['decode_post'])
    targets = {
        'decode_pre': ids1,
        'decode_pre_mask': mask1,
        'decode_post': ids2,
        'decode_post_mask': mask2,
    }
    return features, targets

  def get_training_input(self):
    return self._get_input(self.input_pattern, 'training_input')

  def get_eval_input(self):
    return self._get_input(self.validation_input_pattern, 'eval_input')

  def get_predict_input(self):
    return self._get_input(self.predict_input_pattern, 'predict_input',
                           num_epochs=1, shuffle=False)

  def get_serving_input(self):
    # TODO: Return placeholders for input at serving time
    return {
        'encode': tf.VarLenFeature(dtype=tf.int64),
    }
  def _get_sequence_lengths(self, tensor):
    if isinstance(tensor, tf.SparseTensor):
      _, _, counts = tf.unique_with_counts(tensor.indices[:,0])
      return counts
    else:
      return tf.reduce_sum(tensor, 1)

  def get_predictions(self, features, targets, mode, params):
    """Build and return the model."""

    if mode == tf.contrib.learn.ModeKeys.INFER:
      # At inference time, we don't care about decoding again.
      encode_input = features['encode']
      if not isinstance(encode_input, tf.Tensor):
        encode_input, _ = self._sparse_to_batch(encode_input)
      word_emb = tf.get_variable(
          'word_embedding',
          shape=[self.vocab_size, params['embedding_dim']],
          initializer=self.uniform_initializer)
      encode_emb = tf.nn.embedding_lookup(word_emb, encode_input)
      encode_lengths = tf.to_int32(self._get_sequence_lengths(encode_input), name='length')
      thought_vectors = self._setup_encoder(encode_emb, encode_lengths)
      return {
          'thought_vectors': thought_vectors,
      }

    encode_input = features['encode']
    decode_pre_input = targets['decode_pre']
    decode_post_input = targets['decode_post']
    tf.summary.histogram('inputs/encode', encode_input)
    tf.summary.histogram('inputs/decode_pre', decode_pre_input)
    tf.summary.histogram('inputs/decode_post', decode_post_input)

    # Share the word embedding between encoder and decoder.
    word_emb = tf.get_variable(
        'word_embedding',
        shape=[self.vocab_size, params['embedding_dim']],
        initializer=self.uniform_initializer)
    tf.summary.histogram('word_embedding', word_emb)

    encode_emb = tf.nn.embedding_lookup(word_emb, encode_input)
    encode_lengths = tf.to_int32(self._get_sequence_lengths(encode_input), name='length')
    # Now encode a thought vector and feed it into the two decoders.
    thought_vectors = self._setup_encoder(encode_emb, encode_lengths)
    # When training, feed the thought vector into two separate
    # networks: predicting the previous and next sentences
    decode_pre_input = tf.nn.embedding_lookup(word_emb, decode_pre_input)
    decode_pre_mask = targets['decode_pre_mask']
    decode_pre_lengths = self._get_sequence_lengths(decode_pre_mask)
    decode_pre = self._setup_decoder(
        'decode_pre', thought_vectors,
        decode_pre_input, decode_pre_lengths,
        reuse=False)
    decode_post_input = tf.nn.embedding_lookup(word_emb, decode_post_input)
    decode_post_mask = targets['decode_post_mask']
    decode_post_lengths = self._get_sequence_lengths(decode_post_mask)
    decode_post = self._setup_decoder(
        'decode_post', thought_vectors,
        decode_post_input, decode_post_lengths,
        reuse=True)
    return {
        'decode_pre_logits': decode_pre,
        'decode_pre_mask': decode_pre_mask,
        'decode_post_logits': decode_post,
        'decode_post_mask': decode_post_mask,
    }

  def get_loss(self, predictions, targets, mode, params):
    if mode != tf.contrib.learn.ModeKeys.INFER:
      decode_pre_loss = self._decoder_loss(
          'decode_pre', predictions['decode_pre_logits'],
          predictions['decode_pre_mask'], targets['decode_pre'])
      decode_post_loss = self._decoder_loss(
          'decode_post', predictions['decode_post_logits'],
          predictions['decode_post_mask'], targets['decode_post'])
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
          embeddings[:, :-1, :], [[0, 0], [1, 0], [0, 0]], name='input')
      decoder_output, _ = tf.nn.dynamic_rnn(
          cell=cell,
          inputs=decoder_input,
          sequence_length=sequence_lengths,
          initial_state=initial_state,
          scope=scope)
    tf.summary.histogram('rnn/'+name, decoder_output)

    # Stack batch vertically.
    decoder_output = tf.reshape(decoder_output, [-1, num_units])

    # Logits.
    with tf.variable_scope('logits', reuse=reuse) as scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=decoder_output,
          num_outputs=self.vocab_size,
          activation_fn=None,
          weights_initializer=self.uniform_initializer,
          scope=scope)
    tf.summary.histogram('logits/'+name, logits)
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
        num_units = self.params['encoder_dim'] // 2
        cell_fw = self._get_cell(num_units)
        cell_bw = self._get_cell(num_units)
        _, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=sequence_lengths,
            dtype=tf.float32,
            scope=scope)
        for i, state in enumerate(states):
          tf.summary.histogram('rnn/state_{}'.format(i), state)
        thought_vectors = tf.concat(states, 1, name='thought_vectors')
      else:
        cell = self._get_cell(self.params['encoder_dim'])
        _, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            sequence_length=sequence_lengths,
            dtype=tf.float32,
            scope=scope)
        thought_vectors = tf.identity(state, name='thought_vectors')
    tf.summary.histogram('thought_vectors', thought_vectors)
    return thought_vectors
