"""
@file: dataset_paris_multi.py
@time: 2018/11/2 15:03
@desc: generate "genuine,genuine,forgery" pairs

"""

import os

import numpy as np
import scipy.io as sio

from utils import contains_any
from dataset.g_feature_extractor import extract_global_feature


def _parse_function(input, label):
    return input, label


def _fix_length(data, length):
    len_data = len(data)
    if len_data >= length:
        return data[:length], length
    shape_append = list(np.shape(data))
    shape_append[0] = length - len_data
    return np.append(data, np.zeros(shape_append), axis=0), len_data


def _stroke_splitter(p):
    stroke_start = 0
    index_no_ink = np.where(p <= 0)[0]
    splitter = []
    for i in range(len(index_no_ink)):
        if index_no_ink[i] - stroke_start <= 2:
            stroke_start = index_no_ink[i]
            continue
        splitter.append((stroke_start, index_no_ink[i]))
        stroke_start = index_no_ink[i] + 1
    if len(p) - stroke_start > 1:
        splitter.append((stroke_start, len(p)))
    return splitter


def _features_mat(file, params, p_name):
    # print('reading: ', file)
    data = sio.loadmat(file)
    input_data = []
    for key in data:
        if not key.startswith('__'):
            values = data[key][0]
            input_data.append(values)

    input_data = np.transpose(input_data, [1, 0])

    length = len(input_data)
    p = data[p_name][0]
    return input_data, _stroke_splitter(p), length


def _normalize_length(points, length_dest):
    """将点的序列正则化为同样的长度,目前只能处理下采样（多变少）"""
    length_src = len(points)
    k = length_src / length_dest
    points_dest = []
    for i in range(length_dest):
        length_features = len(length_src[i])
        point = [0] * length_features
        m_base = int(i * k)
        m_up = int((i + 1) * k)
        for m in range(m_base, m_up):
            for j in range(length_features):
                """mean"""
                point[j] = (point[j] * (m - m_base) +
                            points[m][j]) / (m - m_base + 1)
        points_dest.append(point)
    return points_dest


def _input_from_file(file, params):
    p_name = 'f0'
    input_data, splitter, length = _features_mat(file, params, p_name)
    """为了和手机手写数据集兼容，去除pen-up轨迹"""
    pen_down_points = []
    for start, end in splitter:
        stroke = input_data[start:end]
        stroke[0][0] = 0
        pen_down_points.extend(stroke)
    points = np.array(pen_down_points)
    global_features = extract_global_feature(
        points[:, 1], points[:, 2], points[:, 3], None, normalize=False)

    if not params.stroke_base:
        """ 直接返回原始数据，带有pen-up轨迹"""
        # return input_data, length
        points, length = _fix_length(
            pen_down_points, params.max_sequence_length)
        return points, len(points), None, None, global_features

    strokes = []
    len_per_stroke = []
    strokes_features = []
    for start, end in splitter:
        stroke = input_data[start:end]
        stroke_global_feature = extract_global_feature(
            input_data[:, 1], input_data[:, 2], input_data[:, 3], None, normalize=False)
        stroke, len_stroke = _fix_length(stroke, params.length_per_stroke)
        strokes.append(stroke)
        len_per_stroke.append(len_stroke)
        strokes_features.append(stroke_global_feature)

    strokes = np.array(strokes)
    len_per_stroke = np.array(len_per_stroke)
    strokes, length = _fix_length(strokes, params.length_per_signature)
    len_per_stroke, _ = _fix_length(
        len_per_stroke, params.length_per_signature)
    strokes_features, _ = _fix_length(
        strokes_features, params.length_per_signature)

    return strokes, length, len_per_stroke, strokes_features, global_features


def _sub_sampling(list0, list1, repeating):
    invert = len(list0) < len(list1)
    lists = [list1, list0] if invert else [list0, list1]
    if repeating > 0:
        lists[0] = lists[0][:len(lists[1]) * repeating]
        temp = []
        for i in range(repeating):
            temp.extend(lists[1])
        lists[1] = temp
    return (lists[1], lists[0]) if invert else (lists[0], lists[1])


def _list_dataset_pair(dir_path, listfile_path, pos_repeating=1, only_label=None, random_forged=False):
    data_pos = []
    data_neg = []
    data_random = []
    file = open(listfile_path)
    for i, line in enumerate(file.readlines()):
        items = line.split(' ')
        label = int(items[-1])
        if only_label is not None and label not in only_label:
            continue
        if not random_forged and label == 2:
            continue

        data_temp = data_pos if label == 1 else (
            data_neg if label == 0 else data_random)
        pair_temp = []
        for j in range(len(items) - 1):
            pair_temp.append(os.path.join(dir_path, items[j]))
        data_temp.append([pair_temp, label])

    file.close()
    np.random.shuffle(data_pos)
    np.random.shuffle(data_neg)
    np.random.shuffle(data_random)
    max_len = 15000
    if len(data_pos) > max_len:
        data_pos = data_pos[:max_len]
    if len(data_neg) > max_len:
        data_neg = data_neg[:max_len]
    if len(data_random) > max_len:
        data_random = data_random[:max_len]

    print(listfile_path, " examples of positive data:  -> %d" % len(data_pos))
    print(listfile_path, " examples of negative data:  -> %d" % len(data_neg))
    print(listfile_path, " examples of random data:  -> %d" % len(data_random))

    if pos_repeating > 0:
        data_pos, data_neg = _sub_sampling(data_pos, data_neg, pos_repeating)
        if len(data_random) > len(data_neg):
            data_random = data_random[:len(data_neg)]
    data_neg.extend(data_pos)
    if random_forged:
        data_neg.extend(data_random)
    data = data_neg
    np.random.shuffle(data)

    return data


def _get_list_pair(params, is_training, pos_repeating=1, only_label=None, only_dataset=None):
    batch_size = params.batch_size * params.num_gpus
    datasets = params.datasets
    list_pair = []
    for dataset in datasets:
        if only_dataset is not None:
            if not contains_any(dataset['signature_train_list'].lower(), only_dataset.split('@')):
                continue
        listfile_path = dataset['signature_train_list'] if is_training else dataset['signature_val_list']
        pairs = _list_dataset_pair(dataset['dir_path'], listfile_path, pos_repeating, only_label,
                                   random_forged=params.random_forged)
        list_pair.extend(pairs)
    np.random.shuffle(list_pair)
    steps = len(list_pair) / batch_size
    print("examples after sub sampling:  -> %d ,steps:-> %d " %
          (len(list_pair), steps))
    return list_pair, len(list_pair)


def _max(iters):
    if len(iters) <= 1:
        return iters[0]
    if len(iters) == 2:
        return np.maximum(iters[0], iters[1])
    else:
        splitter = int(len(iters) / 2)
        return _max([_max(iters[:splitter]), _max(iters[splitter:])])


def generator_pairs(pairs_list, params):
    for pair, label in pairs_list:
        datas = []
        lengths = []
        len_per_strokes = []
        all_strokes_features = []
        all_global_features = []

        for file in pair:
            data, length, len_per_stroke, strokes_features, global_features = _input_from_file(
                file, params)
            datas.append(data)
            lengths.append(length)
            len_per_strokes.append(len_per_stroke)
            all_strokes_features.append(strokes_features)
            all_global_features.append(global_features)

        axis = 2 if params.stroke_base else 1
        all_global_features = np.concatenate(all_global_features, 0).tolist()

        if params.stroke_base:
            all_strokes_features = np.concatenate(
                all_strokes_features, 1).tolist()
            len_per_stroke = np.stack(len_per_strokes, 1).tolist()
            # len_per_stroke = _max(len_per_strokes)
        else:
            len_per_stroke = None
        yield label, np.concatenate(datas, axis), lengths, len_per_stroke, all_strokes_features, all_global_features


def input_fn(params, is_training, pos_repeating=1, only_label=None, only_dataset=None):
    pairs_list, length_dataset = _get_list_pair(params, is_training,
                                                pos_repeating=pos_repeating,
                                                only_label=only_label, only_dataset=only_dataset)
    return generator_pairs(pairs_list, params), int(length_dataset / (params.batch_size * params.num_gpus))


def get_iter(params, is_training, is_loop, only_label=None, only_dataset=None):
    epoch = 1
    repeating = 1 if is_training else -1
    # repeating = -1
    while True:
        data, _ = input_fn(params, is_training=is_training, pos_repeating=repeating, only_label=only_label,
                           only_dataset=only_dataset)

        label = []
        x = []
        length = []
        len_per_stroke = []
        strokes_features = []
        global_features = []
        for item in data:
            label.append(item[0])
            x.append(item[1])
            length.append(item[2])
            len_per_stroke.append(item[3])
            strokes_features.append(item[4])
            global_features.append(item[5])
            if len(label) < params.batch_size * params.num_gpus:
                continue
            yield np.array(label), x, np.array(length), len_per_stroke, strokes_features, global_features
            label = []
            x = []
            length = []
            len_per_stroke = []
            strokes_features = []
            global_features = []
        if is_training:
            epoch = epoch + 1
            print('epoch: ', epoch)
        if not is_loop and (epoch > params.num_epochs or not is_training):
            break
