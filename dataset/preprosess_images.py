# encoding: utf-8

"""
@file: preprosess_images.py
@time: 2018/5/8 18:
@desc: regularize images, binaries, turn into black background

"""
import os
import sys

import imageio
import numpy as np

dir_to_process = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/cfca_clean/'
dir_processed = '/home/deeplearning/work/Deeplearning/dataset/writingID/online/cfca_clean_png_processed/'


def _normalize_images(images_dir, processed_dir, reverse):
    """binaries, turn into black background """
    for root, dirs, files in os.walk(images_dir):
        for name in files:
            new_path = os.path.join(processed_dir, os.path.split(root)[-1])
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            if name.lower().endswith('.jpg'):
                image = imageio.imread(os.path.join(root, name))
                image[np.where(image < 230)] = 0
                image[np.where(image >= 230)] = 255
                if reverse:
                    image = 255 - image
                imageio.imwrite(os.path.join(new_path, name), image)
    print('all images processed!')


def _normalize_cfca_images(images_dir, processed_dir, reverse):
    """binaries by the alpha channel, turn into black background """
    for root, dirs, files in os.walk(images_dir):
        for dir_name in dirs:
            if not dir_name.endswith('_png'):
                continue
            save_dir = os.path.join(root, dir_name)
            save_dir = save_dir.replace(images_dir, processed_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for name in files:
            new_path = root.replace(images_dir, processed_dir)
            if name.lower().endswith('.png'):
                image = imageio.imread(os.path.join(root, name))
                indexs = np.where(image[:, :, 3] < 100)
                if len(indexs[0]) > 10:
                    indexs = np.where(image[:, :, 3] > 100)
                    for i in range(3):
                        image[:, :, i][indexs] = 255
                else:
                    image = 255 - image

                if not reverse:
                    image = 255 - image
                imageio.imwrite(os.path.join(new_path, name.replace('.png', '.jpg')), image[:, :, 0:3])
    print('all images processed!')


def main(argv=None):
    if argv is None:
        argv = sys.argv
    _normalize_cfca_images(dir_to_process, dir_processed, True)


if __name__ == "__main__":
    sys.exit(main())
