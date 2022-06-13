import os
import random

import numpy
from PIL import Image


def load_img_data(path):
    images = os.listdir(path)
    data_list = []
    label_list = []
    for x in images:
        if not (x.startswith('M-')) and not (x.startswith('W-')):
            continue
        sex, pid, sp = x.split('-')
        img = Image.open(path + x)
        img = img.convert('L')
        data = numpy.array(img, 'f')
        data = data.ravel()
        data_list.append(data)
        if sex == 'M':
            label_list.append(pid)
        else:
            label_list.append(str(int(pid) + 50))
    return data_list, label_list


def get_split_index():
    split_list = []
    while len(split_list) < 7:
        x = random.randint(0, 25)
        if split_list.count(x) == 0:
            split_list.append(x)
    split_list.sort()
    for x in range(2, 101):
        for i in range(0, 7):
            split_list.append(split_list[i] + (x - 1) * 26)
    return split_list


def dataset_split(data_list: list, label_list: list, split_list: list):
    x_test = []
    y_test = []
    for x in reversed(split_list):
        x_test.append(data_list.pop(x))
        y_test.append(label_list.pop(x))
    x_train = data_list
    y_train = label_list
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    data_list, label_list = load_img_data('./dataset/')
    split_list = get_split_index()
    x_train, y_train, x_test, y_test = dataset_split(data_list, label_list, split_list)
    print()