import numpy


class Fisher:
    """
    Fisher 判别器类
    """""
    def __init__(self):
        self.w = []
        self.sw = []
        self.data = []
        self.label = []
        self.dataMean = []

    def reshape_dataset(self, x_train, y_train):
        """
        根据标签将同一类的样本进行整合
        
        :param x_train: 训练数据集
        :param y_train: 训练标签集
        :return: 无返回值
        """""
        print('\n正在进行样本聚合……')
        tempX = ''
        tempList = []
        index = 0
        for x in y_train:
            if x == tempX:
                # 标签未发生变化时，继续划为一类
                tempList.append(x_train[index])
                index += 1
            else:
                # 标签发生变化，已有的类加入 data 列表中，并开辟新集合容纳后续标签的样本
                self.data.append(numpy.array(tempList))
                tempList.clear()
                tempList.append(x_train[index])
                index += 1
                self.label.append(x)
                tempX = x
        self.data.append(numpy.array(tempList))
        # 清除第一个空集合
        self.data.pop(0)
        self.data = numpy.array(self.data)
        print('完成了样本聚合\n')

    def get_mean(self):
        """
        计算类中数据的均值
        
        :return: 无返回值
        """""
        print('\n正在进行均值计算……')
        mean = []
        for i in range(len(self.data)):
            mean.clear()
            for k in range(len(self.data[i][0])):
                m = 0
                for j in range(len(self.data[i])):
                    m += self.data[i][j][k]
                m /= len(self.data[i])
                mean.append(m)
            self.dataMean.append(numpy.array(mean))
        self.dataMean = numpy.array(self.dataMean)
        print('完成了均值计算\n')

    def get_sw(self):
        """
        计算类中数据的散度
        
        :return: 无返回值
        """""
        print('\n正在进行散度计算……')
        for x in range(len(self.data)):
            sw_row = []
            for y in range(len(self.data)):
                sw = numpy.zeros((len(self.data[x][0]), len(self.data[x][0])))
                s1 = numpy.zeros((len(self.data[x][0]), len(self.data[x][0])))
                s2 = numpy.zeros((len(self.data[x][0]), len(self.data[x][0])))
                # 相同的两类散度为 0
                if x == y:
                    sw_row.append(sw)
                    continue
                # 计算第一类散度
                for i in range(len(self.data[x])):
                    tempS = numpy.asmatrix(self.data[x][i] - self.dataMean[x])
                    temp = numpy.matmul(tempS.T, tempS)
                    s1 += temp
                # 计算第二类散度
                for j in range(len(self.data[y])):
                    tempS = numpy.asmatrix(self.data[y][j] - self.dataMean[y])
                    temp = numpy.matmul(tempS.T, tempS)
                    s2 += temp
                sw = s1 + s2
                sw_row.append(sw)
            self.sw.append(sw_row)
        print('完成了散度计算\n')

    def get_w(self):
        """
        计算类中数据的映射向量

        :return: 无返回值
        """""
        print('\n正在进行投影向量计算……')
        for x in range(len(self.dataMean)):
            w_row = []
            for y in range(len(self.dataMean)):
                w = []
                if x == y:
                    w_row.append(w)
                else:
                    mean_minuet = self.dataMean[x] - self.dataMean[y]
                    sw = numpy.linalg.inv(self.sw[x][y])
                    w = numpy.dot(sw, mean_minuet)
                    w_row.append(w)
            self.w.append(w_row)
        print('完成了投影向量计算\n')

    def get_range(self, w, samples):
        """
        计算指定数据在进行映射后在一维空间的最小值与最大值
        
        :param w: 指定两类间的映射向量
        :param samples: 指定类的样本集
        :return: 最小值和最大值
        """""
        max = numpy.dot(w, samples[0])
        min = numpy.dot(w, samples[0])
        for x in range(1, len(samples)):
            y = numpy.dot(w, samples[x])
            max = y if y > max else max
            min = y if y < min else min
        return min, max

    def calculate(self, target):
        """
        预测指定数据的标签
        
        :param target: 需要预测的数据
        :return: 标签
        """""
        score = numpy.zeros(len(self.label))
        for x in range(len(self.dataMean)):
            for y in range(len(self.dataMean)):

                # 两个一样标签的数据集间不采用 Fisher 判别
                if x == y:
                    continue

                # 获取两类在一维空间的上下界
                x_min, x_max = self.get_range(self.w[x][y], self.data[x])
                y_min, y_max = self.get_range(self.w[x][y], self.data[y])
                target_value = numpy.dot(self.w[x][y], target)
                # 预测值刚好落在 x 类范围中
                if x_min < target_value < x_max:
                    score[x] += 1
                    continue
                # 预测值刚好落在 y 类范围中
                if y_min < target_value < y_min:
                    score[y] += 1
                    continue
                # 预测值不在两类间的情况
                # x 类范围大于 y 类 范围
                if x_min > y_max:
                    # 预测值在 x 类之上
                    if target_value > x_max:
                        score[x] += 1
                        continue
                    # 预测值在 y 类之下
                    if target_value < y_min:
                        score[y] += 1
                        continue
                    # 预测值在两类之间
                    temp = x if (x_min - target_value) > (target_value - y_max) else y
                    score[temp] += 1
                    continue
                # y 类范围大于 x 类范围
                if y_min > x_max:
                    # 预测值在 y 类之上
                    if target_value > y_max:
                        score[y] += 1
                        continue
                    # 预测值在 x 类之下
                    if target_value < x_min:
                        score[x] += 1
                        continue
                    # 预测值在两类之间
                    temp = y if (y_min - target_value) > (target_value - x_max) else x
                    score[temp] += 1
                    continue
        # 返回置信度最大的标签结果
        return self.label[score.argmax()]

    def predict(self, x_test):
        """
        对测试集进行预测

        :param x_test: 测试数据集
        :return: 预测标签集合
        """""
        predict = []
        for x in range(len(x_test)):
            print('正在进行第' + str(x) + '个目标的检测')
            predict.append(self.calculate(x_test[x]))
        return predict

    def initialization(self, x_train, y_train):
        """
        初始化 Fisher 判别器，计算所有必须的参数
        
        :param x_train: 训练数据集
        :param y_train: 训练标签集
        :return: 无返回值
        """""
        self.reshape_dataset(x_train, y_train)
        self.get_mean()
        self.get_sw()
        self.get_w()

    def calculate_accuracy(self, result_label, y_test):
        """
        计算预测结果的准确度
        
        :param result_label: 预测标签集合
        :param y_test: 测试标签集
        :return: 无返回值
        """""
        max = len(result_label)
        correct = 0
        for x in range(len(result_label)):
            if result_label[x] == y_test[x]:
                correct += 1
        result = correct / max
        print("最终的准确率为：%.5f" % result)
