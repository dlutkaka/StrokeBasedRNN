# encoding: utf-8

"""
@file: processor_biosecurID.py
@time: 2018/7/16 11:40
@desc:

"""

from __future__ import division

import os

import matplotlib.pyplot as plt
import scipy.io as sio

from dataset.data_utils import *

data_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/BiosecurID-SONOF-DB/OnlineReal/'
processed_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/BiosecurID-SONOF-DB/processed_normalized/'

parser = argparse.ArgumentParser()
parser.add_argument('-src_path', default=data_path)
parser.add_argument('-processed_path',
                    default=processed_path)
args = parser.parse_args()

num_genuine = 16
num_forgery = 12
genuine_list = []
forgery_list = []
for i in range(1, 5):
    for j in [1, 2, 6, 7]:
        genuine_list.append('s%04d_sg%04d' % (i, j))
    for j in [3, 4, 5]:
        forgery_list.append('s%04d_sg%04d' % (i, j))


def generate_list(train_size, listfile_name, n_v_1=1, random_fogery=False):
    list_file_train = open(listfile_name + '_train.txt', 'w')
    list_file_test = open(listfile_name + '_val.txt', 'w')

    signers_list = list(range(1, 133))
    np.random.shuffle(signers_list)

    for index, signer in enumerate(signers_list):
        list_file = list_file_train if index < train_size else list_file_test
        genuine_list_signer = ["u1%03d%s%s" % (signer, file, '.mat') for file in genuine_list]
        forgery_list_signer = ["u1%03d%s%s" % (signer, file, '.mat') for file in forgery_list]

        random_forgery_list = None
        if random_fogery:
            """加入random_fogery, 不同名字的签名对"""
            all_signers = signers_list[:train_size] if index < train_size else signers_list[train_size:]
            if index == train_size:
                print(index)
            random_forgery_singer = get_random_signer(5, signer, all_signers)
            random_forgery_list = []
            for forgery_signer in random_forgery_singer:
                random_forgery_list.extend(["u1%03d%s%s" % (forgery_signer, file, '.mat') for file in forgery_list])

        write_pairs(list_file, genuine_list_signer, forgery_list_signer, n_v_1=n_v_1, random_forged=random_forgery_list)

    list_file_train.close()
    list_file_test.close()


def draw_line(x, y, z, order_stroke=0):
    alpha = max([0, 1 - order_stroke / 28])
    z = 255 - ((z / 1023) * 255).astype(np.int32)
    for i in range(1, len(x)):
        # color_hex = '#%02x%02x%02x' % (255, 255, 255)
        color_hex = '#%02x%02x%02x' % (z[i], z[i], z[i])
        plt.plot(x[i - 1:i + 1], y[i - 1:i + 1], color=color_hex, linewidth=3, alpha=1, linestyle='-')


def plot_by_stroke():
    plt.ioff()
    # plt.style.use('dark_background')
    fig = plt.figure(frameon=False, clear=True)
    # fig = plt.figure(frameon=False, clear=True, figsize=(2.20, 1.55), dpi=100)

    max_points = 0
    min_points = 99999
    max_p = 0
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if not file.endswith('mat'):
                continue
            data = sio.loadmat(os.path.join(data_path, file))
            x = data['x'][0]
            y = data['y'][0]
            p = data['p'][0]
            max_points = max(len(x), max_points)
            min_points = min(len(x), min_points)
            max_p = max(max_p, max(p))

            draw_index = 0
            order_stroke = 0
            index_no_ink = np.where(p <= 0)[0]

            num_stroke = 0

            for i in range(len(index_no_ink)):
                if index_no_ink[i] - draw_index <= 1:
                    draw_index = index_no_ink[i]
                    continue
                # draw_line(x[draw_index:index_no_ink[i]], y[draw_index:index_no_ink[i]],
                #           p[draw_index:index_no_ink[i]], order_stroke)
                print('stroke_length', index_no_ink[i] - draw_index)
                draw_index = index_no_ink[i]
                order_stroke = order_stroke + 1
                num_stroke = num_stroke + 1

            if len(x) - draw_index > 1:
                # draw_line(x[draw_index:], y[draw_index:],
                #           p[draw_index:], order_stroke)
                num_stroke = num_stroke + 1
            print('\nnum_stroke', num_stroke)

            file_name = os.path.join(processed_path, file.replace('.mat', '.png'))

            plt.axis('off')
            fig.savefig(file_name, bbox_inches='tight', transparent=False, pad_inches=0)
            fig.clear()

    print('max_p', max_p)
    print('max_points', max_points)
    print('min_points', min_points)


def extend_database(path, save_path):
    path_exp = os.path.expanduser(path)
    all_files = [path for path in os.listdir(path_exp)
                 if os.path.isfile(os.path.join(path_exp, path))]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in all_files:
        if not file.endswith('mat'):
            continue
        data = sio.loadmat(path_exp + file)
        x = data['x'][0].tolist()
        y = data['y'][0].tolist()
        p = data['p'][0].tolist()
        stroke_mark = mark_stroke(p)
        time_functions = cal_time_functions(x, y, p, stroke_mark)
        count = 0
        time_functions_dic = {}
        for time_function in time_functions:
            time_function = np.array(time_function)
            time_functions_dic["f{}".format(count)] = time_function
            count += 1
        save_file = save_path + file
        if not os.path.exists(save_file):
            os.system(r"touch {}".format(save_file))
        sio.savemat(save_file, time_functions_dic)


def generate_feature_map(signature_file, feature_length):
    result = []
    data = sio.loadmat(signature_file)
    x = data['x'][0].tolist()
    y = data['y'][0].tolist()
    p = data['p'][0].tolist()
    stroke_mark = mark_stroke(p)
    time_functions = cal_time_functions(x, y, p, stroke_mark)
    if len(time_functions) <= 0:
        return
    for i in range(feature_length):
        point_feature = []
        for j in range(len(time_functions)):
            if i < len(time_functions[0]):
                point_feature.append(time_functions[j][i])
            else:
                point_feature.append(0)
        result.append(point_feature)
    return result


def main():
    # extend_database(args.src_path, args.processed_path)
    # generate_list(111, '../experiments/datalist/2v1_biosecur_mat', n_v_1=2)
    generate_list(111, './experiments/datalist/1v1_biosecur_mat', n_v_1=1, random_fogery=True)


if __name__ == '__main__':
    main()
