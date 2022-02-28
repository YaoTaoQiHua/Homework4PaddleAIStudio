# -*- coding:utf-8 -*-

"""
@ Author: WeiMiQu
@ Contact: 1830465230@qq.com
@ Date: 2022/2/19 上午11:55 
@ Software: PyCharm
@ File: dataset.py
@ Desc: 
"""

from paddle.vision.datasets import DatasetFolder


class MyImageNetDataset(DatasetFolder):
    def __init__(self,
                 root,
                 loader=None,
                 extensions=None,
                 transform=None,
                 is_valid_file=None):
        super(MyImageNetDataset, self).__init__(root=root,
                                                loader=loader,
                                                extensions=extensions,
                                                transform=transform,
                                                is_valid_file=is_valid_file)

        # self.root = root
        # self.transform = transform


if __name__ == '__main__':

    import os
    import cv2
    import tempfile
    import shutil
    import numpy as np


    def make_fake_dir():
        data_dir = tempfile.mkdtemp()

        for i in range(2):
            sub_dir = os.path.join(data_dir, 'class_' + str(i))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            for j in range(2):
                fake_img = (np.random.random((32, 32, 3)) * 255).astype('uint8')
                cv2.imwrite(os.path.join(sub_dir, str(j) + '.jpg'), fake_img)
        return data_dir


    temp_dir = make_fake_dir()
    # temp_dir is root dir
    # temp_dir/class_1/img1_1.jpg
    # temp_dir/class_2/img2_1.jpg
    data_folder = MyImageNetDataset(temp_dir)

    for items in data_folder:
        break

    shutil.rmtree(temp_dir)

