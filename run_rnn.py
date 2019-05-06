# encoding: utf-8

"""
@file: run_estimator_cnn.py
@time: 2018/8/20 18:11
@desc:

"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

# from rnn.dataset_paris_multi import get_iter
from rnn.dataset_paris import get_iter
from rnn.models import Input
from rnn.models import model_signature
from utils import Params
from utils import compute_eer
from utils import contains_any

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test/')
parser.add_argument('--mode', default='train')
parser.add_argument('--dataset')
parser.add_argument('--steps', default=0, type=int)


def _compute_eer(distance, label, random_forgery=False):
    fogery_label = 2.0 if random_forgery else 0.0
    distance_neg = distance[np.where(np.array(label) == fogery_label)][:, 0]
    distance_pos = distance[np.where(np.array(label) == 1.0)][:, 0]
    eer, threshold = compute_eer(
        distance_positive=distance_pos, distance_negative=distance_neg)
    return eer


def main():
    args = parser.parse_args()

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    json_path = 'dataset/params.json'
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False, gpu_options=gpu_options)
    # config = tf.ConfigProto(allow_soft_placement=True)
    config.intra_op_parallelism_threads = 4  # No. physical cores
    config.inter_op_parallelism_threads = 4  # No. physical cores
    # config.gpu_options.allow_growth = True

    input_placeholder = Input(params)
    loss, stroke_loss, distance, sec_at_spe, merged_summary = model_signature(input_placeholder,
                                                                              tf.estimator.ModeKeys.TRAIN,
                                                                              params)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.exponential_decay(params.learning_rate,
                                               global_step=global_step,
                                               decay_steps=600, decay_rate=0.6)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(params.learning_rate)

    """TransferLearning"""
    var_list = tf.trainable_variables()
    load_var_list = [
        var for var in var_list if 'Attention_layer' not in var.name]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            loss, global_step=global_step, colocate_gradients_with_ops=True)

    saver = tf.train.Saver(max_to_keep=30, allow_empty=True)
    t_vars = tf.local_variables()
    local_vars = [var for var in t_vars if
                  contains_any(var.name,
                               ['true_positives', 'true_negatives', 'false_positives', 'false_negatives', 'total',
                                'count'])]

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_train = tf.summary.FileWriter(
            args.model_dir + '/train', sess.graph)
        summary_val = tf.summary.FileWriter(
            args.model_dir + '/val', sess.graph)
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt:
            # transfer_saver = tf.train.Saver(var_list=load_var_list)
            # transfer_saver.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restored model %s" % ckpt)

        log_val = "Validate: step: %d, loss: %.5f, stroke_loss: %.5f, eer: %.5f, accuracy: %.5f "
        log_train = "step: %d,train_time: %4.4f, loss: %.5f, stroke_loss: %.5f,  eer: %.5f, accuracy: %.5f "
        start_time = time.time()

        if args.mode.lower() == 'train':
            """save the config file"""
            shutil.copy(json_path, args.model_dir + 'params.json')
            data_val = get_iter(params, is_training=False,
                                is_loop=True, only_dataset=args.dataset)
            data_train = get_iter(params, is_training=True,
                                  is_loop=False, only_dataset=args.dataset)
            for item in data_train:
                tf.variables_initializer(local_vars).run()
                label, x, length, len_per_stroke, stroke_global_feature, global_features = item
                _, loss_v, stroke_loss_v, distance_v, sec_at_spe_v, merged_summary_v, step = sess.run(
                    [train_op, loss, stroke_loss, distance, sec_at_spe, merged_summary, global_step], feed_dict={
                        input_placeholder.x: x,
                        input_placeholder.y: label,
                        input_placeholder.length: length,
                        input_placeholder.len_per_stroke: len_per_stroke,
                        input_placeholder.strokes_features: stroke_global_feature,
                        input_placeholder.global_features: global_features,
                        input_placeholder.is_training: True
                    })

                eer = _compute_eer(distance_v, label)
                eer_scalar = tf.Summary(
                    value=[tf.Summary.Value(tag='eer', simple_value=eer)])
                summary_train.add_summary(merged_summary_v, step)
                summary_train.add_summary(eer_scalar, step)
                if step % 10 == 0:
                    train_time = time.time() - start_time
                    print(log_train % (step, train_time, loss_v,
                                       stroke_loss_v, eer, sec_at_spe_v))
                    start_time = time.time()

                if step % params.save_checkpoints_steps == 0 and step > 0:
                    saver.save(sess, args.model_dir +
                               'signet', global_step=step)

                if step % 100 == 1:
                    distances = []
                    labels = []
                    for i in range(39):
                        tf.variables_initializer(local_vars).run()
                        label, x, length, len_per_stroke, stroke_global_feature, global_features = next(
                            data_val)
                        loss_v, stroke_loss_v, distance_v, sec_at_spe_v, merged_summary_v, step = sess.run(
                            [loss, stroke_loss, distance, sec_at_spe,
                                merged_summary, global_step],
                            feed_dict={
                                input_placeholder.x: x,
                                input_placeholder.y: label,
                                input_placeholder.length: length,
                                input_placeholder.len_per_stroke: len_per_stroke,
                                input_placeholder.strokes_features: stroke_global_feature,
                                input_placeholder.global_features: global_features,
                                input_placeholder.is_training: False
                            })
                        distances.append(distance_v)
                        labels.append(label)
                    distances = np.concatenate(distances)
                    labels = np.concatenate(labels)
                    eer = _compute_eer(distances, labels)
                    eer_scalar = tf.Summary(
                        value=[tf.Summary.Value(tag='eer', simple_value=eer)])
                    print(log_val %
                          (step, loss_v, stroke_loss_v, eer, sec_at_spe_v))
                    summary_val.add_summary(merged_summary_v, step)
                    summary_val.add_summary(eer_scalar, step)

            saver.save(sess, args.model_dir + 'signet', global_step=step)

        if args.mode.lower() == 'evaluate':
            params.batch_size = params.batch_size * 10
            data_val = get_iter(params, is_training=False, is_loop=False, only_label=[0, 1],
                                only_dataset=args.dataset)
            distance_all = []
            label_all = []
            eval_step = 0
            start_time = time.time()
            for item in data_val:
                label, x, length, len_per_stroke, stroke_global_feature, global_features = item
                loss_v, distance_v, sec_at_spe_v, merged_summary_v, step = sess.run(
                    [loss, distance, sec_at_spe, merged_summary, global_step],
                    feed_dict={
                        input_placeholder.x: x,
                        input_placeholder.y: label,
                        input_placeholder.length: length,
                        input_placeholder.len_per_stroke: len_per_stroke,
                        input_placeholder.strokes_features: stroke_global_feature,
                        input_placeholder.global_features: global_features,
                        input_placeholder.is_training: False
                    })
                distance_all.append(distance_v)
                label_all.append(label)
                eer = _compute_eer(distance_v, label)
                print(log_val % (eval_step, loss_v, 0, eer, sec_at_spe_v))
                eval_step += 1
                if 0 < args.steps <= eval_step:
                    break

            distance_all = np.concatenate(distance_all, axis=0)
            label_all = np.concatenate(label_all, axis=0)
            eer_skilled = _compute_eer(distance_all, label_all)
            eer_random = 0.0
            if params.random_forged:
                eer_random = _compute_eer(
                    distance_all, label_all, random_forgery=True)
            print('\neer of evalution is skilled : %.5f  random : %.5f, time: %4.4f' % (
                eer_skilled, eer_random, time.time() - start_time))


if __name__ == "__main__":
    sys.exit(main())
