# encoding: utf-8

"""
@file: processor_mobisig.py
@time: 2018/7/16 11:40
@desc:

"""
import os
import sys

import matplotlib.pyplot as plt
import pandas
import scipy.io as sio

from dataset.data_utils import *

mobisig_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/mobisig/'
processed_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/mobisig_processed/'
max_azimuth_angle = 389
max_altitude_angel = 90
min_azimuth_angle = -360
min_altitude_angel = 22
max_strokes = 28


def draw_line(x, y, z, angle_az, angle_in, order_stroke=0):
    alpha = max([0, 1 - order_stroke / 28])
    z = ((z / 1023) * 255).astype(np.int32)
    angle_az = (((angle_az - min_azimuth_angle) / (max_azimuth_angle - min_azimuth_angle)) * 255).astype(np.int32)
    angle_in = (((angle_in - min_altitude_angel) / (max_altitude_angel - min_altitude_angel)) * 255).astype(np.int32)
    for i in range(1, len(x)):
        color_hex = '#%02x%02x%02x' % (z[i], z[i], z[i])
        plt.plot(x[i - 1:i + 1], y[i - 1:i + 1], color=color_hex, linewidth=3, alpha=1, linestyle='-')


def draw_png(argv=None):
    if argv is None:
        argv = sys.argv
    plt.ioff()
    plt.style.use('dark_background')
    fig = plt.figure(frameon=False, clear=True)

    if not os.path.exists(processed_path):
        os.mkdir(processed_path)
    """TODO"""


def _features_file(file):
    data = pandas.read_csv(file)
    x = np.array(data.x)
    y = np.array(data.y)
    t = np.array(data.timestamp)
    velocityx = np.array(data.velocityx)
    velocityx[np.where(velocityx != 0)] = 1
    stroke_mark = velocityx

    if x is None:
        return None, None, None, None, None
    x_y = np.concatenate([np.reshape(x, [len(x), 1]), np.reshape(y, [len(y), 1])], axis=1)
    distance = np.sqrt(np.sum(np.square(x_y[1:] - x_y[:-1]), axis=1))
    t_diff = t[1:] - t[:-1]
    t_diff = np.reshape(t_diff + 1, [len(t_diff)])
    z = distance / t_diff
    z = np.insert(z, 0, z[0], axis=0)
    index_stroke = np.where(stroke_mark == 0)[0]
    z = np.append(z, [0], axis=0)
    z[index_stroke] = z[index_stroke + 1]
    return x, y, z[:-1], t, stroke_mark


def convert2mat(data_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for root, dirs, files in os.walk(data_path):
        for dir_name in dirs:
            save_dir = os.path.join(root, dir_name)
            save_dir = save_dir.replace(data_path, processed_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for file in files:
            if not file.endswith('.csv') or file == 'users.csv':
                continue
            filepath = os.path.join(root, file)
            x, y, p, t, stroke_mark = _features_file(filepath)
            if x is None:
                continue
            t = None
            time_functions = cal_time_functions(x, y, p, stroke_mark, time=t)
            count = 0
            time_functions_dic = {}
            for time_function in time_functions:
                time_function = np.array(time_function)
                time_functions_dic["f{}".format(count)] = time_function
                count += 1

            save_dir = root.replace(data_path, processed_path)
            save_file = os.path.join(save_dir, file.replace('.csv', '.mat'))
            if not os.path.exists(save_file):
                os.system(r"touch {}".format(save_file))
            sio.savemat(save_file, time_functions_dic)


def generate_list(train_size, data_path, listfile_name, n_v_1=1, random_fogery=False):
    list_file_train = open(listfile_name + '_train.txt', 'w')
    list_file_test = open(listfile_name + '_val.txt', 'w')
    signers_list = os.listdir(data_path)
    np.random.shuffle(signers_list)

    for index, signer in enumerate(signers_list):
        list_file = list_file_train if index <= train_size else list_file_test
        files = os.listdir(os.path.join(data_path, signer))
        genuine_files = [os.path.join(signer, file) for file in files if file.startswith('SIGN_GEN_')]
        forged_files = [os.path.join(signer, file) for file in files if file.startswith('SIGN_FOR_')]

        random_forgery_list = None
        if random_fogery:
            all_signers = signers_list[:train_size] if index < train_size else signers_list[train_size:]
            if index == train_size:
                print(index)
            random_forgery_singer = get_random_signer(5, signer, all_signers)
            random_forgery_list = []
            for forgery_signer in random_forgery_singer:
                files = os.listdir(os.path.join(data_path, forgery_signer))
                random_forgery_list.extend(
                    [os.path.join(forgery_signer, file) for file in files if file.startswith('SIGN_FOR_')])

        write_pairs(list_file, genuine_files, forged_files, n_v_1, random_forged=random_forgery_list)

    list_file_train.close()
    list_file_test.close()
    print('finish generating pair list!!!')


def main():
    # convert2mat(mobisig_path, processed_path)
    data_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/mobisig_processed/'
    # generate_list(70, data_path, './experiments/datalist/2v1_mobisig_sample', n_v_1=2)
    generate_list(70, data_path, './experiments/datalist/1v1_mobisig', n_v_1=1, random_fogery=True)


if __name__ == "__main__":
    sys.exit(main())
