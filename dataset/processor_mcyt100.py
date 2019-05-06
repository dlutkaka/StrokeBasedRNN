# encoding: utf-8

"""
@file: processor_mcyt100.py
@time: 2018/7/16 11:40
@desc:

"""
import os
import struct
import sys

import matplotlib.pyplot as plt
import scipy.io as sio

from dataset.data_utils import *

mcyt_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/mcyt100'
processed_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/mcyt100_processed/'
# processed_path = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/mcyt100_gray'
# mcyt_path = 'C:\work\Projects\HWS_ID\dataset\MCTY_Signature_100/'
# processed_path = 'C:\work\Projects\HWS_ID\dataset\MCTY_Signature_100_processed/'
max_azimuth_angle = 389
max_altitude_angel = 90
min_azimuth_angle = -360
min_altitude_angel = 22
max_strokes = 28

num_genuine = 24
num_forged = 24


def generate_list(train_size, listfile_name, n_v_1=1, random_fogery=False):
    signers_list = list(range(0, 100))
    list_file_train = open(listfile_name + '_train.txt', 'w')
    list_file_test = open(listfile_name + '_val.txt', 'w')

    train_indexs = np.arange(0, len(signers_list), 1)
    np.random.shuffle(train_indexs)
    train_indexs = train_indexs[:train_size]

    for index, signer in enumerate(signers_list):
        list_file = list_file_train if index in train_indexs else list_file_test
        genuine_list = list(range(1, num_genuine + 1))
        forgery_list = list(range(1, num_forged + 1))
        genuine_list_signer = ["%04d%s%04d%s%02d%s" % (signer, '/', signer, 'v', file, '.mat') for file in genuine_list]
        forgery_list_signer = ["%04d%s%04d%s%02d%s" % (signer, '/', signer, 'f', file, '.mat') for file in forgery_list]

        random_forgery_list = None
        if random_fogery:
            all_signers = signers_list[:train_size] if index < train_size else signers_list[train_size:]
            if index == train_size:
                print(index)
            random_forgery_singer = get_random_signer(5, signer, all_signers)
            random_forgery_list = []
            for forgery_signer in random_forgery_singer:
                random_forgery_list.extend(
                    ["%04d%s%04d%s%02d%s" % (forgery_signer, '/', forgery_signer, 'f', file, '.mat') for file in
                     forgery_list])

        write_pairs(list_file, genuine_list_signer, forgery_list_signer, n_v_1=n_v_1, random_forged=random_forgery_list)

    list_file_train.close()
    list_file_test.close()


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
    # fig = plt.figure(frameon=False, clear=True, figsize=(2.20, 1.55), dpi=100)

    max_az = 0
    max_in = 0
    min_az = 3600
    min_in = 900
    max_stroke_num = 0
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)
    for root, dirs, files in os.walk(mcyt_path):
        for file in files:
            if not file.endswith('fpg'):
                continue
            file_bin = open(os.path.join(root, file), 'rb')
            id = file_bin.read(4)
            hsize = int.from_bytes(file_bin.read(2), byteorder='little')
            ver = 2 if (hsize == 48 or hsize == 60) else 1
            format = int.from_bytes(file_bin.read(2), byteorder='little')
            if not format == 4:
                continue
            m = int.from_bytes(file_bin.read(2), byteorder='little')
            can = int.from_bytes(file_bin.read(2), byteorder='little')
            ts = int.from_bytes(file_bin.read(4), byteorder='little')
            res = int.from_bytes(file_bin.read(2), byteorder='little')
            file_bin.seek(4, 1)
            coef = int.from_bytes(file_bin.read(4), byteorder='little')
            mvector = int.from_bytes(file_bin.read(4), byteorder='little')
            nvectores = int.from_bytes(file_bin.read(4), byteorder='little')
            nc = int.from_bytes(file_bin.read(2), byteorder='little')

            if ver == 2:
                Fs = int.from_bytes(file_bin.read(4), byteorder='little')
                mventana = int.from_bytes(file_bin.read(4), byteorder='little')
                msolapadas = int.from_bytes(file_bin.read(4), byteorder='little')
            file_bin.seek(hsize - 12, 0)
            dator = int.from_bytes(file_bin.read(4), byteorder='little')
            delta = int.from_bytes(file_bin.read(4), byteorder='little')
            ddelta = int.from_bytes(file_bin.read(4), byteorder='little')
            file_bin.seek(hsize, 0)
            res = int(res / 8)
            if not res == 4:
                print(file, ': res:', res)

            tam_tot = nvectores * can * mvector * res
            temp = file_bin.read(tam_tot)
            h = 0
            data_matrix = np.zeros((nvectores, mvector), dtype=np.float32)
            for i in range(0, nvectores):
                for m in range(0, mvector):
                    data_matrix[i, m] = struct.unpack_from('<f', temp[h:h + res])[0]
                    h = h + res
            # print(data_matrix)
            index_no_ink = np.where(data_matrix[:, 2:3] == 0)[0]

            image_path = root.replace(mcyt_path.split('\\')[-1], processed_path.split('\\')[-1])
            file_name = os.path.join(image_path, file.replace('.fpg', '.png'))
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            x = data_matrix[:, :1].astype(np.int32)
            y = data_matrix[:, 1:2].astype(np.int32)
            x = (x - np.min(x)).reshape([len(x)])
            y = (y - np.min(y)).reshape([len(y)])

            z = data_matrix[:, 2:3].reshape([len(x)])
            angle_az = data_matrix[:, 3:4].reshape([len(x)])
            angle_in = data_matrix[:, 4:5].reshape([len(x)])
            max_az = max([max_az, max(angle_az)])
            max_in = max([max_in, max(angle_in)])
            min_az = min([min_az, min(angle_az)])
            min_in = min([min_in, min(angle_in)])

            draw_index = 0
            order_stroke = 0

            for i in range(len(index_no_ink)):
                if index_no_ink[i] - draw_index <= 1:
                    draw_index = index_no_ink[i]
                    continue
                draw_line(x[draw_index:index_no_ink[i]], y[draw_index:index_no_ink[i]],
                          z[draw_index:index_no_ink[i]], angle_az[draw_index:index_no_ink[i]],
                          angle_in[draw_index:index_no_ink[i]], order_stroke)
                draw_index = index_no_ink[i]
                order_stroke = order_stroke + 1
            max_stroke_num = max(max_stroke_num, order_stroke)

            if len(x) - draw_index > 1:
                draw_line(x[draw_index:], y[draw_index:],
                          z[draw_index:], angle_az[draw_index:],
                          angle_in[draw_index:], order_stroke)

            plt.axis('off')
            fig.savefig(file_name, bbox_inches='tight', transparent=False, pad_inches=0)
            fig.clear()
    plt.close(fig)

    print(max_az, max_in, min_az, min_in, max_strokes)


def _features_file(file):
    file_bin = open(file, 'rb')
    id = file_bin.read(4)
    hsize = int.from_bytes(file_bin.read(2), byteorder='little')
    ver = 2 if (hsize == 48 or hsize == 60) else 1
    format = int.from_bytes(file_bin.read(2), byteorder='little')
    if not format == 4:
        return None, None, None, None, None
    m = int.from_bytes(file_bin.read(2), byteorder='little')
    can = int.from_bytes(file_bin.read(2), byteorder='little')
    ts = int.from_bytes(file_bin.read(4), byteorder='little')
    res = int.from_bytes(file_bin.read(2), byteorder='little')
    file_bin.seek(4, 1)
    coef = int.from_bytes(file_bin.read(4), byteorder='little')
    mvector = int.from_bytes(file_bin.read(4), byteorder='little')
    nvectores = int.from_bytes(file_bin.read(4), byteorder='little')
    nc = int.from_bytes(file_bin.read(2), byteorder='little')

    if ver == 2:
        Fs = int.from_bytes(file_bin.read(4), byteorder='little')
        mventana = int.from_bytes(file_bin.read(4), byteorder='little')
        msolapadas = int.from_bytes(file_bin.read(4), byteorder='little')
    file_bin.seek(hsize - 12, 0)
    dator = int.from_bytes(file_bin.read(4), byteorder='little')
    delta = int.from_bytes(file_bin.read(4), byteorder='little')
    ddelta = int.from_bytes(file_bin.read(4), byteorder='little')
    file_bin.seek(hsize, 0)
    res = int(res / 8)
    if not res == 4:
        print(file, ': res:', res)

    tam_tot = nvectores * can * mvector * res
    temp = file_bin.read(tam_tot)
    h = 0
    data_matrix = np.zeros((nvectores, mvector), dtype=np.float32)
    for i in range(0, nvectores):
        for m in range(0, mvector):
            data_matrix[i, m] = struct.unpack_from('<f', temp[h:h + res])[0]
            h = h + res

    x = data_matrix[:, :1].astype(np.int32)
    y = data_matrix[:, 1:2].astype(np.int32)
    x = x.reshape([len(x)])
    y = y.reshape([len(y)])
    z = data_matrix[:, 2:3].reshape([len(x)])

    stroke_mark = mark_stroke(z)
    if x is None:
        return None, None, None, None, None
    return x, y, z, None, stroke_mark


def extend_database(data_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for root, dirs, files in os.walk(data_path):
        for dir_name in dirs:
            save_dir = os.path.join(root, dir_name)
            save_dir = save_dir.replace(data_path, processed_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for file in files:
            if not file.endswith('.fpg'):
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
            save_file = os.path.join(save_dir, file.replace('.fpg', '.mat'))
            if not os.path.exists(save_file):
                os.system(r"touch {}".format(save_file))
            sio.savemat(save_file, time_functions_dic)


def main():
    # extend_database(mcyt_path, processed_path)
    # generate_list(85, './experiments/datalist/2v1_mcyt', n_v_1=2)
    generate_list(85, './experiments/datalist/1v1_mcyt', n_v_1=1, random_fogery=True)


if __name__ == "__main__":
    sys.exit(main())
