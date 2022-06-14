from PIL import Image
import os
import numpy
import random


class ImgProcess:
    """
    图片加载与处理类
    """""
    def __init__(self, path):
        self.path = path
        self.data_list = []
        self.label_list = []
        self.split_list = []
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

    def load_img_data(self):
        """
        加载指定目录中的图片，并根据文件名创建标签，并对图片进行灰度化处理，结果作为数据集
        :return: 无返回值
        """""
        images = os.listdir(self.path)
        data_list = []
        label_list = []
        for x in images:
            if not (x.startswith('M-')) and not (x.startswith('W-')):
                continue
            sex, pid, sp = x.split('-')
            img = Image.open(self.path + x)
            img = img.convert('L')
            data = numpy.array(img, 'f')
            data = data.ravel()
            data_list.append(data)
            if sex == 'M':
                label_list.append(pid)
            else:
                label_list.append(str(int(pid) + 50))
        self.data_list = data_list
        self.label_list = label_list

    def get_split_index(self):
        """
        对每个人随机选取 7 张照片作为训练集
        :return: 无返回值
        """""
        split_list = []
        while len(split_list) < 7:
            x = random.randint(0, 25)
            if split_list.count(x) == 0:
                split_list.append(x)
        split_list.sort()
        for x in range(2, 101):
            for i in range(0, 7):
                split_list.append(split_list[i] + (x - 1) * 26)
        self.split_list = split_list

    def dataset_split(self):
        """
        进行数据集的分割
        :return: 无返回值
        """""
        x_test = []
        y_test = []
        for x in reversed(self.split_list):
            x_test.append(self.data_list.pop(x))
            y_test.append(self.label_list.pop(x))
        self.x_train = numpy.array(self.data_list) / 255
        self.y_train = self.label_list
        self.x_test = numpy.array(x_test) / 255
        self.y_test = y_test

    def run(self):
        """
        进行图片加载分割操作
        :return: 训练数据与标签集，测试数据与标签集
        """""
        self.load_img_data()
        self.get_split_index()
        self.dataset_split()
        return self.x_train, self.y_train, self.x_test, self.y_test

