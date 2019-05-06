# encoding: utf-8

"""
@file: autoencoder.py
@time: 2019/2/13 11:06
@desc: 

"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope

from rnn.attention_model import attention

rnn_cell = tf.nn.rnn_cell.LSTMCell


def stroke_encode(x, len_per_stroke, scope, is_training, stroke_feature=None, convolution=False, bidirectional=True,
                  is_attention=True, stroke_rnn_hidden=[200, 100]):
    with variable_scope.variable_scope(
            scope, 'SigRnn', reuse=tf.AUTO_REUSE):
        origin_shape = x.shape
        inputs = tf.reshape(x, [-1, 400, 24])
        len_per_stroke = tf.reshape(len_per_stroke, [-1])
        if convolution:
            inputs, _ = _add_conv_layers(inputs, is_training=is_training)

        keep_prob = tf.cond(is_training, lambda: 0.65, lambda: 1.0)
        fw_cells = _get_rnn_cells(
            stroke_rnn_hidden, keep_prob, name='signature_cell_fw')
        bw_cells = _get_rnn_cells(
            stroke_rnn_hidden, keep_prob, name='signature_cell_bw')

        len_per_stroke = tf.cast(len_per_stroke, tf.int32)
        if bidirectional:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cells, cell_bw=fw_cells,
                                                                     inputs=inputs,
                                                                     sequence_length=len_per_stroke, dtype=tf.float32,
                                                                     scope='bid_stroke_rnn')
            # outputs = tf.concat(outputs, axis=2)

            if is_attention:  # and 1 == 2:
                with tf.name_scope('Attention_stroke_layer'):
                    outputs, alphas = attention(
                        outputs, 200, return_alphas=True)
                    tf.summary.histogram('alphas', alphas)
            else:
                outputs = tf.concat(
                    [output_states[0][-1].h, output_states[-1][-1].h], axis=1)
        else:
            outputs, output_states = tf.nn.dynamic_rnn(cell=fw_cells, inputs=inputs,
                                                       sequence_length=len_per_stroke,
                                                       dtype=tf.float32, scope='stroke_rnn')
            outputs = output_states[-1]

        # mask = tf.tile(
        #     tf.expand_dims(tf.sequence_mask(len_per_stroke, tf.shape(outputs)[1]), 2),
        #     [1, 1, tf.shape(outputs)[2]])
        # zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
        # outputs = tf.reduce_sum(zero_outside, axis=1)

        outputs_embedding = outputs
        if stroke_feature is not None:
            stroke_feature = tf.reshape(stroke_feature, [-1, 61])
            outputs_embedding = tf.concat(
                [outputs_embedding, stroke_feature], 1)

        """decoder"""
        decoder_input = tf.zeros_like(inputs)
        fw_cells_decoder = _get_rnn_cells(
            stroke_rnn_hidden, keep_prob, name='signature_cell_fw_decoder')
        bw_cells_decoder = _get_rnn_cells(
            stroke_rnn_hidden, keep_prob, name='signature_cell_bw_decoder')
        dec_outputs, dec_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cells_decoder,
                                                                 cell_bw=bw_cells_decoder,
                                                                 inputs=decoder_input,
                                                                 sequence_length=len_per_stroke,
                                                                 initial_state_fw=output_states[0],
                                                                 initial_state_bw=output_states[1],
                                                                 scope='bid_stroke_rnn_decoder')
        dec_outputs = tf.concat(dec_outputs, axis=2)
        # dec_outputs = dec_outputs[::-1]
        dec_weight = tf.get_variable(name='dec_weight', shape=[stroke_rnn_hidden[-1] * 2, 24], dtype=tf.float32,
                                     trainable=True)
        dec_bias = tf.get_variable(name='dec_bias', initializer=tf.constant(
            0.1, shape=[24], dtype=tf.float32))
        dec_weight = tf.tile(tf.expand_dims(dec_weight, 0),
                             [8 * 20, 1, 1])
        decoder_output = tf.matmul(dec_outputs, dec_weight) + dec_bias
        # loss = tf.reduce_mean(tf.square(inputs - decoder_output))
        """MAPE(Mean Absolute Percentage Error)"""
        loss = tf.reduce_mean(
            tf.abs(tf.divide(tf.subtract(decoder_output, inputs), (inputs + 1e-10))))

        size_embedding = stroke_rnn_hidden[-1] * \
            2 if bidirectional else stroke_rnn_hidden[-1]
        if stroke_feature is not None:
            size_embedding = size_embedding + 61

        outputs_embedding = tf.reshape(
            outputs_embedding, [-1, 20, size_embedding])
        outputs_embedding = tf.nn.relu(outputs_embedding)

    return outputs_embedding, loss


class LSTMAutoencoder(object):
    """Basic version of LSTM-autoencoder.
  (cf. http://arxiv.org/abs/1502.04681)
  Usage:
    ae = LSTMAutoencoder(hidden_num, inputs)
    sess.run(ae.train)
  """

    def __init__(
            self,
            hidden_num,
            inputs,
            cell=None,
            optimizer=None,
            reverse=True,
            decode_without_input=False,
    ):
        """
    Args:
      hidden_num : number of hidden elements of each LSTM unit.
      inputs : a list of input tensors with size
              (batch_num x elem_num)
      cell : an rnn cell object (the default option
            is `tf.python.ops.rnn_cell.LSTMCell`)
      optimizer : optimizer for rnn (the default option is
              `tf.train.AdamOptimizer`)
      reverse : Option to decode in reverse order.
      decode_without_input : Option to decode without input.
    """

        self.batch_num = inputs[0].get_shape().as_list()[0]
        self.elem_num = inputs[0].get_shape().as_list()[1]

        if cell is None:
            self._enc_cell = rnn_cell(hidden_num)
            self._dec_cell = rnn_cell(hidden_num)
        else:
            self._enc_cell = cell
            self._dec_cell = cell

        with tf.variable_scope('encoder'):
            (self.z_codes, self.enc_state) = tf.contrib.rnn.static_rnn(
                self._enc_cell, inputs, dtype=tf.float32)

        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num,
                                                           self.elem_num], dtype=tf.float32), name='dec_weight'
                                      )
            dec_bias_ = tf.Variable(tf.constant(0.1,
                                                shape=[self.elem_num],
                                                dtype=tf.float32), name='dec_bias')

            if decode_without_input:
                dec_inputs = [tf.zeros(tf.shape(inputs[0]),
                                       dtype=tf.float32) for _ in
                              range(len(inputs))]
                (dec_outputs, dec_state) = tf.contrib.rnn.static_rnn(self._dec_cell, dec_inputs,
                                                                     initial_state=self.enc_state,
                                                                     dtype=tf.float32)
                if reverse:
                    dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0,
                                                                   2])
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0),
                                      [self.batch_num, 1, 1])
                self.output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_
            else:

                dec_state = self.enc_state
                dec_input_ = tf.zeros(tf.shape(inputs[0]),
                                      dtype=tf.float32)
                dec_outputs = []
                for step in range(len(inputs)):
                    if step > 0:
                        vs.reuse_variables()
                    (dec_input_, dec_state) = \
                        self._dec_cell(dec_input_, dec_state)
                    dec_input_ = tf.matmul(dec_input_, dec_weight_) \
                        + dec_bias_
                    dec_outputs.append(dec_input_)
                if reverse:
                    dec_outputs = dec_outputs[::-1]
                self.output_ = tf.transpose(tf.stack(dec_outputs), [1,
                                                                    0, 2])

        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.loss = tf.reduce_mean(tf.square(self.input_
                                             - self.output_))

        if optimizer is None:
            self.train = tf.train.AdamOptimizer().minimize(self.loss)
        else:
            self.train = optimizer.minimize(self.loss)


def _add_conv_layers(convolved, is_training):
    """Adds convolution layers."""
    num_conv = [64, 48, 32]
    conv_len = [5, 3, 3, 3, 3]
    poll_len = [1, 1, 1, 1, 1, 2]
    poll_stride = [1, 1, 1, 1, 2, 2]
    batch_norm = False
    for i in range(len(num_conv)):
        if batch_norm:
            convolved = tf.layers.batch_normalization(
                convolved,
                training=is_training)
        convolved = tf.layers.conv1d(
            convolved,
            filters=num_conv[i],
            kernel_size=conv_len[i],
            activation=None,
            strides=1,
            padding="same",
            name="conv1d_%d" % i,
            reuse=tf.AUTO_REUSE)
        if i == 5:
            convolved = tf.layers.dropout(
                convolved,
                rate=0.5,
                training=is_training)
        convolved = tf.nn.relu(convolved)
        if poll_len[i] > 1:
            convolved = tf.layers.average_pooling1d(
                convolved, [poll_len[i]], poll_stride[i], name="poll1d_%d" % i)
    convolved = tf.layers.dropout(
        convolved,
        rate=0.5,
        training=is_training)
    convolved = tf.nn.relu(convolved)
    return convolved, length_seq(convolved)


def length_seq(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def _get_rnn_cells(num_hidden, keep_prob, name=None):
    cells = []
    for i in range(len(num_hidden)):
        cell = rnn_cell(num_hidden[i], name=(
            name + str(i) if name is not None else name))
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=keep_prob, state_keep_prob=keep_prob)
        # cell = tf.contrib.rnn.DropoutWrapper(cell)
        cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells=cells)
