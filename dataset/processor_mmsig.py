# encoding: utf-8

"""
@file: processor_mmsig.py
@time: 2018/7/16 11:40
@desc:

"""
import os
import sys

import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image

from dataset.data_utils import *

data_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/BIP-MMSIG/mobile/'
temp_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/BIP-MMSIG_temp/mobile/'
processed_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/BIP-MMSIG_processed/mobile/'
global_max_z = 100
global_max_strokes = 11

parser = argparse.ArgumentParser()
parser.add_argument('-src_path', default=data_path)
parser.add_argument('-processed_path',
                    default=processed_path)
args = parser.parse_args()

num_genuine = 16
num_forgery = 12
genuine_list = list(range(1, 21))
forgery_list = list(range(21, 41))


def generate_list(train_size, listfile_name, n_v_1=1, random_fogery=False):
    list_file_train = open(listfile_name + '_train.txt', 'w')
    list_file_test = open(listfile_name + '_val.txt', 'w')

    signers_list = list(range(1, 57 + 1))
    signers_list.remove(32)
    signers_list.remove(33)
    np.random.shuffle(signers_list)

    str_format = 'U%02dS%d%s'
    for index, signer in enumerate(signers_list):
        list_file = list_file_train if index <= train_size else list_file_test

        genuine_list_signer = [str_format % (signer, file, '.mat') for file in genuine_list]
        forgery_list_signer = [str_format % (signer, file, '.mat') for file in forgery_list]
        random_forgery_list = None
        if random_fogery:
            all_signers = signers_list[:train_size] if index < train_size else signers_list[train_size:]
            if index == train_size:
                print(index)
            random_forgery_singer = get_random_signer(5, signer, all_signers)
            random_forgery_list = []
            for forgery_signer in random_forgery_singer:
                random_forgery_list.extend([str_format % (forgery_signer, file, '.mat') for file in forgery_list])

        write_pairs(list_file, genuine_list_signer, forgery_list_signer, n_v_1=n_v_1, random_forged=random_forgery_list)

    list_file_train.close()
    list_file_test.close()


def draw_line(x, y, z, order_stroke=0):
    alpha = max([0, 1 - order_stroke / global_max_strokes])
    z[np.where(z > global_max_z)] = 100
    z = ((z / global_max_z) * 255).astype(np.int32)
    z = 255 - z
    for i in range(1, len(x)):
        # color_hex = '#%02x%02x%02x' % (z[i], z[i], z[i])
        # color_hex = '#FFFFFF'
        color_hex = '#000000'
        plt.plot(x[i - 1:i + 1], y[i - 1:i + 1], color=color_hex, linewidth=3, alpha=1, linestyle='-')


def convert_png():
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)
    for root, dirs, files in os.walk(temp_path):
        for file in files:
            if not file.endswith('png'):
                continue
            im = Image.open(os.path.join(root, file))
            im_flip = im.transpose(Image.FLIP_TOP_BOTTOM)
            im_flip = im_flip.convert('RGB')
            im_flip.save(os.path.join(processed_path, file.replace('.png', '.jpg')))


def draw_online_data():
    plt.ioff()
    # plt.style.use('dark_background')
    fig = plt.figure(frameon=False, clear=True)
    # fig = plt.figure(frameon=False, clear=True, figsize=(2.20, 1.55), dpi=100)

    max_z = 0
    max_stroke_num = 0
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if not file.endswith('txt'):
                continue
            file_bin = open(os.path.join(root, file), 'r')
            # for line in file_bin.readlines():
            #     line.replace('\n', '').split(' ')
            data_matrix = [line.replace('\n', '').split(' ') for line in file_bin.readlines()]
            data_matrix = np.array(data_matrix).astype(np.int32)
            index_stroke = np.where(data_matrix[:, 3:4] == 0)[0]
            x = data_matrix[:, :1].astype(np.int32)
            y = data_matrix[:, 1:2].astype(np.int32)
            x_y = data_matrix[:, 0:2].astype(np.int32)
            t = data_matrix[:, 2:3].astype(np.int32)
            distance = np.sqrt(np.sum(np.square(x_y[1:] - x_y[:-1]), axis=1))
            t_diff = t[1:] - t[:-1]
            t_diff = np.reshape(t_diff, [len(t_diff)])
            z = distance / t_diff
            z = np.insert(z, 0, 0, axis=0)
            z[index_stroke] = 0
            max_z = max([max_z, max(z)])

            ax = plt.gca()
            ax.invert_yaxis()
            for i in range(0, len(index_stroke)):
                stroke_start = index_stroke[i]
                stroke_end = index_stroke[i + 1] if i < (len(index_stroke) - 1) else len(x)
                draw_line(x[stroke_start:stroke_end], y[stroke_start:stroke_end],
                          z[stroke_start:stroke_end], i)

            max_stroke_num = max(max_stroke_num, len(index_stroke))

            file_name = os.path.join(temp_path, file.replace('.txt', '.png'))
            plt.axis('off')
            fig.savefig(file_name, bbox_inches='tight', transparent=False, pad_inches=0)
            fig.clear()
            print('max_z: ', max(z), 'stroke_num: ', len(index_stroke))

    plt.close(fig)
    print(max_z, max_stroke_num)


def _features_file(file):
    file_bin = open(file, 'r')
    data_matrix = [line.replace('\n', '').split(' ') for line in file_bin.readlines()]
    file_bin.close()
    length = len(data_matrix)
    data_matrix = np.array(data_matrix).astype(np.int32)
    x = np.reshape(data_matrix[:, 0:1], [length])
    y = np.reshape(data_matrix[:, 1:2], [length])
    x_y = data_matrix[:, 0:2]
    t = np.reshape(data_matrix[:, 2:3], [length])
    stroke_mark = np.reshape(data_matrix[:, 3:4], [length])
    distance = np.sqrt(np.sum(np.square(x_y[1:] - x_y[:-1]), axis=1))
    t_diff = t[1:] - t[:-1]
    t_diff = np.reshape(t_diff, [len(t_diff)])
    z = distance / t_diff
    z = np.insert(z, 0, z[0], axis=0)
    index_stroke = np.where(stroke_mark == 0)[0]
    z[index_stroke] = z[index_stroke + 1]
    return x, y, z, t, stroke_mark


def extend_database(path, save_path):
    path_exp = os.path.expanduser(path)
    all_files = [path for path in os.listdir(path_exp)
                 if os.path.isfile(os.path.join(path_exp, path))]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in all_files:
        if not file.endswith('.txt'):
            continue
        x, y, p, t, stroke_mark = _features_file(path_exp + file)
        t = None
        time_functions = cal_time_functions(x, y, p, stroke_mark, time=t)
        count = 0
        time_functions_dic = {}
        for time_function in time_functions:
            time_function = np.array(time_function)
            time_functions_dic["f{}".format(count)] = time_function
            count += 1
        save_file = save_path + file.replace('.txt', '.mat')
        if not os.path.exists(save_file):
            os.system(r"touch {}".format(save_file))
        sio.savemat(save_file, time_functions_dic)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    # draw_online_data()
    # extend_database(args.src_path, args.processed_path)
    # generate_list(50, './experiments/datalist/2v1_mmsig', n_v_1=2)
    generate_list(49, './experiments/datalist/1v1_mmsig', n_v_1=1, random_fogery=True)


if __name__ == "__main__":
    sys.exit(main())
