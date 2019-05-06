# encoding: utf-8

"""
@file: data_utils.py
@time: 2018/9/13 9:56
@desc:

"""
import argparse
import copy
import math
import random

import numpy as np
from sklearn.preprocessing import StandardScaler

data_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/BiosecurID-SONOF-DB/OnlineReal/'
processed_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/BiosecurID-SONOF-DB/processed_normalized/'

parser = argparse.ArgumentParser()
parser.add_argument('-src_path', default=data_path)
parser.add_argument('-processed_path',
                    default=processed_path)
args = parser.parse_args()


def combine(l, k):
    # if k <= 1:
    #     return l
    answers = []
    one = [0] * k

    def next_c(li=0, ni=0):
        if ni == k:
            answers.append(copy.copy(one))
            return
        for lj in range(li, len(l)):
            one[ni] = l[lj]
            next_c(lj + 1, ni + 1)

    next_c()
    return answers


def combine_2list(list1, list2):
    answers = []
    for i1 in list1:
        for i2 in list2:
            answers.append([i1, i2])
    return answers


def expand_list(nested_list):
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            for sub_item in expand_list(item):
                yield sub_item
        else:
            yield item


def _write_list(pairs, listfile, label):
    for pair in pairs:
        line = ''
        for item in expand_list(pair):
            line = line + item + ' '
        line = line + str(label) + '\n'
        listfile.write(line)


def write_pairs(list_file, genuine_files, forged_files, n_v_1=1, random_forged=None):
    genuine_genuine_suf = combine(genuine_files, n_v_1 + 1)
    genuine_forged_suf = combine_2list(combine(genuine_files, n_v_1), forged_files)
    np.random.shuffle(genuine_genuine_suf)
    if len(genuine_genuine_suf) > 1000:
        genuine_genuine_suf = genuine_genuine_suf[: 1000]

    # if len(genuine_forged_suf) > max_size:
    #     np.random.shuffle(genuine_forged_suf)
    #     genuine_forged_suf = genuine_forged_suf[: max_size]

    _write_list(genuine_genuine_suf, list_file, 1)
    _write_list(genuine_forged_suf, list_file, 0)

    max_size = len(genuine_forged_suf)
    if random_forged is not None:
        genuine_random_forged_suf = combine_2list(combine(genuine_files, n_v_1), random_forged)
        if len(genuine_random_forged_suf) > max_size:
            np.random.shuffle(genuine_random_forged_suf)
            genuine_random_forged_suf = genuine_random_forged_suf[: max_size]
        _write_list(genuine_random_forged_suf, list_file, 2)


def get_random_signer(nums_signer, signer, all_signers):
    others = all_signers.copy()
    others.remove(signer)
    random_signer = random.sample(others, nums_signer)  # [0] * nums_signer
    return random_signer


def standard_transform(data):
    # return data
    scaler = StandardScaler()
    data = np.array(data)
    data = data.reshape(-1, 1)
    scaler.fit(data)
    data = scaler.transform(data)
    data = data.reshape(-1)
    return data.tolist()


def derivative(discrete_data):
    length = len(discrete_data)
    if length < 2:
        return
    else:
        result = []
        index = 2
        while index < length - 2:
            der = discrete_data[index + 1] - discrete_data[index - 1] + 2 * (
                discrete_data[index + 2] - discrete_data[index - 2])
            der = der / 10
            result.append(der)
            index += 1

        der0 = discrete_data[1] - discrete_data[0]
        if length > 2:
            der1 = discrete_data[2] - discrete_data[1]
        else:
            der1 = der0
        result.insert(0, der1)
        result.insert(0, der0)

        if length > 3:
            der3 = discrete_data[length - 1] - discrete_data[length - 2]
            der4 = der3
            result.insert(len(result), der3)
            result.insert(len(result), der4)
        elif length == 3:
            result.insert(2, result[1])

        np_result = np.gradient(discrete_data)
        np_result1 = np.gradient(discrete_data, 2)
        np_result2 = np.gradient(discrete_data, 3)

        return np_result


def cal_path_tangent_angle(x_der, y_der):
    angles = map(lambda a_b: 0 if abs(a_b[0]) < 1e-15 else math.atan(a_b[1] / a_b[0]), zip(x_der, y_der))
    return list(angles)


def cal_velocity(x_der, y_der):
    velocities = map(lambda a_b: math.sqrt(a_b[0] * a_b[0] + a_b[1] * a_b[1]), zip(x_der, y_der))
    return list(velocities)


def cal_curvature_radius(angles, velocities):
    angle_der = derivative(angles)
    curvature_radius = map(
        lambda a_b: 0 if abs(a_b[0]) < 1e-15 or int(a_b[1]) == 0 else math.log10(abs(a_b[1] / a_b[0])),
        zip(angle_der, velocities))
    return list(curvature_radius)


def cal_acceleration(angles, velocities):
    velocity_der = derivative(velocities)
    angle_der = derivative(angles)
    c = map(lambda a_b: a_b[0] * a_b[1], zip(velocities, angle_der))
    accelerations = map(lambda a_b: math.sqrt(a_b[0] * a_b[0] + a_b[1] * a_b[1]), zip(velocity_der, list(c)))
    return list(accelerations)


def cal_ratio_of_speed(velocities):
    window_len = 5
    result = []
    length = len(velocities)
    window = velocities[:(window_len if length > window_len else length)]
    window_len = len(window)
    for i in range(window_len):
        j = i
        while j > 0 and window[j] > window[j - 1]:
            temp = window[j - 1]
            window[j - 1] = window[j]
            window[j] = temp
            j -= 1
    result.append(0 if abs(window[window_len - 1]) < 1e-15 else window[0] / window[window_len - 1])

    index = window_len
    while index < length:
        removed = velocities[index - window_len]
        temp = velocities[index]
        find_removed = False
        j = 0
        while j < window_len:
            if window[j] == removed and not find_removed:
                window[j] = temp
                find_removed = True

            if not find_removed:  
                if j < window_len and temp > window[j]:
                    t = window[j]
                    window[j] = temp
                    temp = t
            else:  
                if j + 1 < window_len and window[j + 1] > window[j]:
                    temp = window[j + 1]
                    window[j + 1] = window[j]
                    window[j] = temp

            j += 1

        result.append(0 if abs(window[window_len - 1]) < 1e-15 else window[0] / window[window_len - 1])
        index += 1

    for k in range(length - len(result)):
        result.append(result[len(result) - 1])

    return result


def cal_angle(x, y):
    result = [0]
    for i in range(len(x)):
        if i + 1 < len(x):
            cosine_value = (x[i] * x[i + 1] + y[i] * y[i + 1]) / \
                           ((math.sqrt(x[i] * x[i] + y[i] * y[i])) *
                            (math.sqrt(x[i + 1] * x[i + 1] + y[i + 1] * y[i + 1])))
            if cosine_value > 1:
                cosine_value = 1
            angle = math.acos(cosine_value)
            result.append(angle)
    return result


def cal_sine_value(angel):
    sine_value = map(lambda a: math.sin(a), angel)
    return list(sine_value)


def cal_cosine_value(angle):
    cosine_value = map(lambda a: math.cos(a), angle)
    return list(cosine_value)


def cal_ratio_of_aspect(x, y, step):
    result = []
    for i in range(len(x)):
        if i + step < len(x):
            ratio = 0 if abs(x[i + step] - x[i]) < 1e-15 else abs(y[i + step] - y[i]) / abs(x[i + step] - x[i])
            result.append(ratio)
    value_append = result[- 1] if len(result) > 0 else 0
    for j in range(len(x) - len(result)):
        result.append(value_append)
    return result


def mark_stroke(p):
    stroke_mark = np.zeros(np.shape(p), dtype=np.float32)
    stroke_mark[np.where(np.array(p) > 0)] = 1.0
    return stroke_mark


def cal_time_functions(x, y, p, stroke_mark, time=None):
    time_functions = []
    x_der = derivative(x)
    y_der = derivative(y)
    p_der = derivative(p)

    tangent_angles = cal_path_tangent_angle(x_der, y_der)
    velocities = cal_velocity(x_der, y_der)

    curvature_radiuses = cal_curvature_radius(tangent_angles, velocities)
    accelerations = cal_acceleration(tangent_angles, velocities)
    tangent_angle_der = derivative(tangent_angles)
    velocity_der = derivative(velocities)
    curvature_radius_der = derivative(curvature_radiuses)
    acceleration_der = derivative(accelerations)
    x_der_der = derivative(x_der)
    y_der_der = derivative(y_der)
    velocity_ratio = cal_ratio_of_speed(velocities)
    angels = cal_angle(x, y)
    angels_der = derivative(angels)
    sine_values = cal_sine_value(angels)
    cosine_values = cal_cosine_value(angels)
    aspect_ratio_by_step5 = cal_ratio_of_aspect(x, y, 5)
    aspect_ratio_by_Steps7 = cal_ratio_of_aspect(x, y, 7)

    time_functions.append(stroke_mark)
    if time is not None:
        time_functions.append(time / 2500)
    time_functions.append(standard_transform(x))
    time_functions.append(standard_transform(y))
    time_functions.append(standard_transform(p))
    time_functions.append(standard_transform(tangent_angles))
    time_functions.append(standard_transform(velocities))
    time_functions.append(standard_transform(curvature_radiuses))
    time_functions.append(standard_transform(accelerations))
    time_functions.append(standard_transform(x_der))
    time_functions.append(standard_transform(y_der))
    time_functions.append(standard_transform(p_der))
    time_functions.append(standard_transform(tangent_angle_der))
    time_functions.append(standard_transform(velocity_der))
    time_functions.append(standard_transform(curvature_radius_der))
    time_functions.append(standard_transform(acceleration_der))
    time_functions.append(standard_transform(x_der_der))
    time_functions.append(standard_transform(y_der_der))
    time_functions.append(standard_transform(velocity_ratio))
    time_functions.append(standard_transform(angels))
    time_functions.append(standard_transform(angels_der))
    time_functions.append(standard_transform(sine_values))
    time_functions.append(standard_transform(cosine_values))
    time_functions.append(standard_transform(aspect_ratio_by_step5))
    time_functions.append(standard_transform(aspect_ratio_by_Steps7))

    return time_functions
