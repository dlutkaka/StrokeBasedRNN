# encoding: utf-8

"""
@file: models.py
@time: 2018/8/7 15:03
@desc: rnn models:siamese,2logits

"""

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import variable_scope

import utils
from rnn.attention_model import attention
from rnn.autoencoder import stroke_encode

# rnn_cell = tf.nn.rnn_cell.GRUCell
rnn_cell = tf.nn.rnn_cell.LSTMCell

ATTENTION_SIZE = 20
# WHOLE_RNN_HIDDEN = [150, 100, 100]
WHOLE_RNN_HIDDEN = [100, 50]
# SIGN_RNN_HIDDEN = [150, 100]
STROKE_RNN_HIDDE = [150, 100]
SIGN_RNN_HIDDEN = [150, 100]


# WHOLE_RNN_HIDDEN = [150, 100, 50]
# STROKE_RNN_HIDDE = [150, 100, 100, 50]


def length_seq(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def _rnn(x, length, scope, is_training, bidirectional=True):
    with variable_scope.variable_scope(
            scope, 'SigRnn', reuse=tf.AUTO_REUSE):
        num_hidden = [100]
        keep_prob = tf.cond(is_training, lambda: 0.8, lambda: 1.0)
        fw_cell0 = rnn_cell(num_hidden[0], name='signature_cell_fw0')
        fw_cell0 = tf.contrib.rnn.DropoutWrapper(
            fw_cell0, output_keep_prob=keep_prob)

        fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=[fw_cell0])
        length = tf.squeeze(length)

        if bidirectional:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell_m, cell_bw=fw_cell_m,
                                                                     inputs=x,
                                                                     sequence_length=length, dtype=tf.float32,
                                                                     scope='bid_stroke_rnn')
            outputs = tf.concat(outputs, axis=2)
            # outputs = tf.concat([output_states[0][-1].h, output_states[-1][-1].h], axis=1)
        else:
            outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=fw_cell_m, inputs=x,
                                                             sequence_length=length,
                                                             dtype=tf.float32, scope='stroke_rnn')
            # outputs = encoder_final_state[-1]
        outputs = tf.nn.relu(outputs)
        return outputs


def _loss_siamese(inputs, labels, params):
    inputs = tf.split(inputs, 2, 2)
    input0 = inputs[0]
    input1 = inputs[1]

    out0, state0 = _rnn(input0, 0.8, "side", 46)
    out1, state1 = _rnn(input1, 0.8, "side", 46)

    out = tf.concat([out0, out1], 2)
    out, state = _rnn(out, 0.8, "concat", 23)

    out = tf.sigmoid(state[-1])
    logits = layers_lib.fully_connected(
        out, 2, scope='fc1', reuse=tf.AUTO_REUSE)

    logits_array = tf.split(logits, 2, 1)
    logits_diff = tf.subtract(logits_array[0], logits_array[1])
    if labels is not None:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.int64)))
        return loss, logits_diff
    else:
        return None, logits_diff


def _rnn_stroke_base(x, length, len_per_stroke, scope, is_training, convolution=False, bidirectional=False,
                     is_attention=False):
    with variable_scope.variable_scope(
            scope, 'SigRnn', reuse=tf.AUTO_REUSE):
        origin_shape = x.shape
        inputs = tf.reshape(x, [-1, origin_shape[-2], origin_shape[-1]])
        len_per_stroke = tf.reshape(len_per_stroke, [-1])
        if convolution:
            inputs, _ = _add_conv_layers(inputs, is_training=is_training)

        keep_prob = tf.cond(is_training, lambda: 0.8, lambda: 1.0)
        fw_cells = _get_rnn_cells(
            STROKE_RNN_HIDDE, keep_prob, name='signature_cell_fw')
        bw_cells = _get_rnn_cells(
            STROKE_RNN_HIDDE, keep_prob, name='signature_cell_bw')

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
                        outputs, ATTENTION_SIZE * 4, return_alphas=True)
                    tf.summary.histogram('alphas', alphas)
            else:
                outputs = tf.concat(
                    [output_states[0][-1].h, output_states[-1][-1].h], axis=1)
        else:
            outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=fw_cells, inputs=inputs,
                                                             sequence_length=len_per_stroke,
                                                             dtype=tf.float32, scope='stroke_rnn')
            outputs = encoder_final_state[-1]

        size_embedding = STROKE_RNN_HIDDE[-1] * \
            2 if bidirectional else STROKE_RNN_HIDDE[-1]
        outputs = tf.reshape(outputs, [-1, origin_shape[1], size_embedding])
        outputs = tf.nn.relu(outputs)

        return outputs

        strokes_embeddings = outputs
        fw_stroke_cell = rnn_cell(150, name='signature_cell_fw')
        fw_stroke_cell = tf.contrib.rnn.DropoutWrapper(
            fw_stroke_cell, output_keep_prob=keep_prob)
        fw_stroke_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=[fw_stroke_cell])
        bw_stroke_cell = rnn_cell(150, name='signature_cell_bw')
        bw_stroke_cell = tf.contrib.rnn.DropoutWrapper(
            bw_stroke_cell, output_keep_prob=keep_prob)
        bw_stroke_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=[bw_stroke_cell])

        if bidirectional:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_stroke_cell_m, cell_bw=fw_stroke_cell_m,
                                                                     inputs=strokes_embeddings,
                                                                     sequence_length=length, dtype=tf.float32,
                                                                     scope='bid_signature_layer_0')
            outputs = tf.concat(outputs, axis=2)
        else:
            outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=fw_stroke_cell_m, inputs=strokes_embeddings,
                                                             sequence_length=length,
                                                             dtype=tf.float32, scope='signature_layer_0')
    return outputs


def _add_conv_layers(inks, is_training):
    """Adds convolution layers."""
    convolved = inks
    #
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
        if i > 1 and i % 7 == 0:
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


def _loss_2logits(inputs, labels, length, len_per_stroke, stroke_features, global_features, params, is_training):
    keep_prob = tf.cond(is_training, lambda: 0.7, lambda: 1.0)
    length = tf.cast(length, tf.int32)
    stroke_loss = 0
    if params.early_concat:
        length = tf.reduce_max(length, axis=1)
        if params.stroke_base:
            len_per_stroke = tf.reduce_max(len_per_stroke, axis=2)
            sign_rnn_input = _rnn_stroke_base(inputs, length, len_per_stroke, "stroke_embedding", is_training=is_training,
                                              convolution=params.convolution, bidirectional=params.bidirectional)
        else:
            if params.convolution:
                inputs, length = _add_conv_layers(
                    inputs, is_training=is_training)
            sign_rnn_input = _rnn(
                inputs, length, "side", is_training=is_training, bidirectional=params.bidirectional)
            # sign_rnn_input = inputs
    else:
        axis_split = 3 if params.stroke_base else 2
        inputs = tf.split(inputs, 2, axis_split)
        lengths = tf.split(length, 2, 1)
        length0 = lengths[0]
        length1 = lengths[1]
        input0 = inputs[0]
        input1 = inputs[1]
        if params.stroke_encoder or params.stroke_base:
            len_per_strokes = tf.split(len_per_stroke, 2, 2)
            len_per_stroke0 = len_per_strokes[0]
            len_per_stroke1 = len_per_strokes[1]

        if params.stroke_encoder:
            stroke_features0 = None
            stroke_features1 = None
            if params.global_feature:
                per_stroke_features = tf.split(stroke_features, 2, 2)
                stroke_features0 = per_stroke_features[0]
                stroke_features1 = per_stroke_features[1]
            out0, loss0 = stroke_encode(input0, len_per_stroke0, "stroke_encoder",
                                        stroke_feature=stroke_features0,
                                        is_training=is_training,
                                        stroke_rnn_hidden=STROKE_RNN_HIDDE)
            out1, loss1 = stroke_encode(input1, len_per_stroke1, "stroke_encoder",
                                        stroke_feature=stroke_features1,
                                        is_training=is_training,
                                        stroke_rnn_hidden=STROKE_RNN_HIDDE)
            stroke_loss = tf.reduce_mean(
                [tf.reduce_mean(loss0), tf.reduce_mean(loss1)])

        elif params.stroke_base:
            out0 = _rnn_stroke_base(input0, length, len_per_stroke0, "stroke_embedding", is_training=is_training,
                                    convolution=params.convolution, bidirectional=params.bidirectional)
            out1 = _rnn_stroke_base(input1, length, len_per_stroke1, "stroke_embedding", is_training=is_training,
                                    convolution=params.convolution, bidirectional=params.bidirectional)
        else:
            if params.convolution:
                input0, length0 = _add_conv_layers(
                    input0, is_training=is_training)
                input1, length1 = _add_conv_layers(
                    input1, is_training=is_training)
            # out0 = input0
            # out1 = input1
            out0 = _rnn(input0, length0, "side", is_training=is_training,
                        bidirectional=params.bidirectional)
            out1 = _rnn(input1, length1, "side", is_training=is_training,
                        bidirectional=params.bidirectional)

        sign_rnn_input = tf.concat([out0, out1], 2)
        length = tf.reduce_max(length, axis=1)

    sign_rnn_input = tf.layers.dropout(
        sign_rnn_input, rate=0.5, training=is_training)

    if params.stroke_base and not params.stroke_encoder and params.global_feature:
        stroke_features = tf.layers.dense(stroke_features, 50)
        stroke_features = tf.nn.relu(stroke_features)
        sign_rnn_input = tf.concat([sign_rnn_input, stroke_features], 2)

    num_logits = 3 if params.random_forged else 2
    with variable_scope.variable_scope(
            'concat_layer', 'SigRnn', reuse=tf.AUTO_REUSE):
        final_cells = SIGN_RNN_HIDDEN if params.stroke_base else WHOLE_RNN_HIDDEN
        cell_m = _get_rnn_cells(final_cells, keep_prob)
        if params.bidirectional:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_m, cell_bw=cell_m,
                                                                     inputs=sign_rnn_input,
                                                                     sequence_length=length, dtype=tf.float32,
                                                                     scope='bid_signature_layer_1')
            if params.attention:
                with tf.name_scope('Attention_layer'):
                    outputs, alphas = attention(
                        outputs, ATTENTION_SIZE, return_alphas=True)
                    tf.summary.histogram('alphas', alphas)

            else:
                outputs = tf.concat(outputs, axis=2)
                # outputs = tf.concat([output_states[0][-1].h, output_states[-1][-1].h], axis=1)
        else:
            outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=cell_m, inputs=sign_rnn_input,
                                                             sequence_length=length,
                                                             dtype=tf.float32, scope='signature_layer_1')
        #
        if not params.attention:
            mask = tf.tile(
                tf.expand_dims(tf.sequence_mask(
                    length, tf.shape(outputs)[1]), 2),
                [1, 1, tf.shape(outputs)[2]])
            zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
            outputs = tf.reduce_sum(zero_outside, axis=1)

        outputs = tf.nn.relu(outputs)
        outputs = tf.layers.dropout(outputs, rate=0.4, training=is_training)

        if params.global_feature:
            """加入global features"""
            global_features_hidden = tf.layers.dense(global_features, 50)
            global_features_hidden = tf.nn.relu(global_features_hidden)
            outputs = tf.concat([outputs, global_features_hidden],  1)
            outputs = tf.layers.dense(outputs, 100)
            outputs = tf.layers.dropout(
                outputs, rate=0.8, training=is_training)

        logits = tf.layers.dense(outputs, num_logits)

    logits_array = tf.split(logits, num_logits, 1)
    # logits_diff = tf.add(logits_array[0], logits_array[2]) if params.random_forged else logits_array[0]
    logits_diff = logits_array[0]
    logits_diff = tf.subtract(logits_diff, logits_array[1])
    # logits_diff = tf.subtract(0.0, logits_array[1])
    # logits_diff = logits_array[0] + logits_array[2]
    prediction = tf.argmax(input=logits, axis=1)

    stroke_loss = stroke_loss * 1e-3
    if labels is not None:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.int64)))
        if params.stroke_encoder:
            loss = tf.reduce_sum([loss, stroke_loss], 0)
        return loss, stroke_loss, logits_diff, prediction
    else:
        return None, stroke_loss, logits_diff, prediction


def _loss_2channels(logits, labels):
    """<Learning to Compare Image Patches via Convolutional Neural Networks>"""
    """ convert y from {0,1,2} to {-1,1}"""
    labels_coefficient = tf.where(
        tf.equal(labels, 2.0), tf.ones_like(labels) * -1, labels)
    labels_coefficient = tf.where(tf.equal(labels_coefficient, 0.0), tf.ones_like(labels) * -1,
                                  labels_coefficient)
    labels_coefficient = tf.reshape(labels_coefficient, [-1, 1])
    loss = tf.maximum(0.0, tf.subtract(
        10.0, tf.multiply(labels_coefficient, logits)))
    return tf.reduce_mean(loss)


def _contrastive_loss(y, d, batch_size):
    tmp = y * tf.square(d)
    # tmp= tf.mul(y,tf.square(d))
    tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
    return tf.reduce_sum(tmp + tmp2) / batch_size / 2


def _normlize_distance(distance):
    """normalization of distance"""
    max_val = tf.reduce_max(distance)
    min_val = tf.reduce_min(distance)
    distance_norm = tf.div(tf.subtract(distance, min_val),
                           tf.subtract(max_val, min_val))
    return distance_norm


class Input:
    def __init__(self, params):
        self.y = tf.placeholder(tf.float32, [None])
        self.length = tf.placeholder(tf.float32, [None, 2])
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        if params.global_feature:
            self.global_features = tf.placeholder(tf.float32,
                                                  [None,  params.global_feature_num*2])
            self.strokes_features = tf.placeholder(tf.float32,
                                                   [None, params.length_per_signature, params.global_feature_num*2])
        else:
            self.global_features = tf.placeholder(tf.float32,
                                                  [None,  None])
            self.strokes_features = tf.placeholder(tf.float32,
                                                   [None, None])
        if params.stroke_base:
            self.x = tf.placeholder(tf.float32,
                                    [None, params.length_per_signature, params.length_per_stroke, params.features])
            self.len_per_stroke = tf.placeholder(
                tf.float32, [None, params.length_per_signature, 2])

        else:
            self.x = tf.placeholder(tf.float32,
                                    [None, params.max_sequence_length, params.features])
            self.len_per_stroke = tf.placeholder(tf.float32, [None])


def model_signature(input_placeholder, mode, params):
    features = input_placeholder.x
    labels = input_placeholder.y
    length = input_placeholder.length
    len_per_stroke = input_placeholder.len_per_stroke
    is_training = input_placeholder.is_training
    strokes_features = input_placeholder.strokes_features
    global_features = input_placeholder.global_features

    loss_function = _loss_2logits
    # loss_function = _loss_siamese

    losses_all_tower = []
    stroke_losses_all_tower = []
    distance_all_tower = []
    prediction_all_tower = []
    if params.stroke_base:
        features = tf.reshape(features,
                              [-1, params.length_per_signature, params.length_per_stroke, params.features])
        len_stroke_tower = tf.split(len_per_stroke, params.num_gpus, axis=0)
    else:
        features = tf.reshape(
            features, [-1, params.max_sequence_length, params.features])
    features = tf.cast(features, tf.float32)
    input_all_tower = tf.split(features, params.num_gpus, axis=0)
    length_all_tower = tf.split(length, params.num_gpus, axis=0)
    strokes_features_tower = tf.split(
        strokes_features, params.num_gpus, axis=0)
    global_features_tower = tf.split(
        global_features, params.num_gpus, axis=0)

    labels_all_tower = [None, None, None]
    if labels is not None:
        labels = tf.cast(labels, tf.float32)
        labels = tf.reshape(labels, [-1])
        labels_all_tower = tf.split(labels, params.num_gpus, axis=0)

    for i in range(params.num_gpus):
        worker_device = '/{}:{}'.format('gpu', i)
        input_tower = input_all_tower[i]

        device_setter = utils.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                params.num_gpus, tf.contrib.training.byte_size_load_fn))
        with tf.device(device_setter):
            len_stroke = None if not params.stroke_base else len_stroke_tower[i]
            loss, stroke_loss, distance, prediction = loss_function(input_tower, labels_all_tower[i],
                                                                    length_all_tower[i], len_stroke,
                                                                    strokes_features_tower[i],
                                                                    global_features_tower[i],
                                                                    params, is_training=is_training)
            if labels_all_tower is not None:
                losses_all_tower.append(loss)
            if stroke_loss is not None:
                stroke_losses_all_tower.append(stroke_loss)
            distance_all_tower.append(distance)
            prediction_all_tower.append(prediction)

    consolidation_device = '/cpu:0'
    with tf.device(consolidation_device):
        distance = tf.concat(distance_all_tower, 0)
        distance = tf.reshape(distance, [-1, 1])
        prediction = tf.concat(prediction_all_tower, 0)
        prediction = tf.reshape(prediction, [-1, 1])
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'distance': distance}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.reduce_mean(losses_all_tower, 0)
        stroke_loss = tf.reduce_mean(stroke_losses_all_tower, 0)
        distance_norm = _normlize_distance(distance)
        labels = tf.reshape(labels, [-1, 1])

        accuracy_ops = tf.metrics.accuracy(labels, prediction)

        labels_2value = tf.where(
            tf.equal(labels, 2.0), tf.zeros_like(labels), labels)
        labels_2value = tf.reshape(labels_2value, [-1, 1])
        labels_reversal = tf.reshape(tf.subtract(tf.cast(1.0, tf.float32), labels_2value),
                                     [-1, 1])  # labels_ = !labels;

        positive_distance = tf.reduce_mean(
            tf.multiply(labels_2value, distance))
        negative_distance = tf.reduce_mean(
            tf.multiply(labels_reversal, distance))
        loss_summary = tf.summary.scalar('loss', loss)
        stroke_loss_summary = tf.summary.scalar('stroke_loss', stroke_loss)
        pos_summary = tf.summary.scalar('positive_distance', positive_distance)
        neg_summary = tf.summary.scalar('negative_distance', negative_distance)

        metric_ops = tf.metrics.auc(
            labels_reversal, distance_norm, name='auc_all')
        auc_summary = tf.summary.scalar('auc', metric_ops[1])
        accuracy_summary = tf.summary.scalar('accuracy', accuracy_ops[1])

        sec_at_spe_metric = tf.metrics.sensitivity_at_specificity(
            labels_reversal, distance_norm, 0.90)

        merged_summary = tf.summary.merge(
            [loss_summary, stroke_loss_summary, pos_summary, neg_summary, auc_summary, accuracy_summary])

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {'evaluation_auc': metric_ops,
                               'accuracy': accuracy_ops,
                               'sec_at_spe': sec_at_spe_metric}
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

        else:
            return loss, stroke_loss, distance, accuracy_ops[1], merged_summary
