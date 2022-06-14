import numpy


class Fisher:
    def __init__(self):
        self.w = []
        self.sw = []
        self.data = []
        self.label = []
        self.dataMean = []

    def reshape_dataset(self, x_train, y_train):
        print('\n正在进行样本聚合……')
        tempX = ''
        tempList = []
        index = 0
        for x in y_train:
            if x == tempX:
                tempList.append(x_train[index])
                index += 1
            else:
                self.data.append(numpy.array(tempList))
                tempList.clear()
                tempList.append(x_train[index])
                index += 1
                self.label.append(x)
                tempX = x
        self.data.append(numpy.array(tempList))
        self.data.pop(0)
        self.data = numpy.array(self.data)
        print('完成了样本聚合\n')

    def get_mean(self):
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
        print('\n正在进行散度计算……')
        for x in range(len(self.data)):
            sw_row = []
            for y in range(len(self.data)):
                sw = numpy.zeros((len(self.data[x][0]), len(self.data[x][0])))
                s1 = numpy.zeros((len(self.data[x][0]), len(self.data[x][0])))
                s2 = numpy.zeros((len(self.data[x][0]), len(self.data[x][0])))
                if x == y:
                    sw_row.append(sw)
                    continue
                for i in range(len(self.data[x])):
                    tempS = numpy.asmatrix(self.data[x][i] - self.dataMean[x])
                    temp = numpy.matmul(tempS.T, tempS)
                    s1 += temp
                for j in range(len(self.data[y])):
                    tempS = numpy.asmatrix(self.data[y][j] - self.dataMean[y])
                    temp = numpy.matmul(tempS.T, tempS)
                    s2 += temp
                sw = s1 + s2
                sw_row.append(sw)
            self.sw.append(sw_row)
        print('完成了散度计算\n')

    def get_w(self):
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
        max = numpy.dot(w, samples[0])
        min = numpy.dot(w, samples[0])
        for x in range(1, len(samples)):
            y = numpy.dot(w, samples[x])
            max = y if y > max else max
            min = y if y < min else min
        return min, max

    def calculate(self, target):
        score = numpy.zeros(len(self.label))
        for x in range(len(self.dataMean)):
            for y in range(len(self.dataMean)):
                if x == y:
                    continue
                x_min, x_max = self.get_range(self.w[x][y], self.data[x])
                y_min, y_max = self.get_range(self.w[x][y], self.data[y])
                target_value = numpy.dot(self.w[x][y], target)
                if x_min < target_value < x_max:
                    score[x] += 1
                    continue
                if y_min < target_value < y_min:
                    score[y] += 1
                    continue
                if x_min > y_max:
                    if target_value > x_max:
                        score[x] += 1
                        continue
                    if target_value < y_min:
                        score[y] += 1
                    temp = x if (x_min - target_value) > (target_value - y_max) else y
                    score[temp] += 1
                    continue
                if y_min > x_max:
                    if target_value > y_max:
                        score[y] += 1
                        continue
                    if target_value < x_min:
                        score[x] += 1
                        continue
                    temp = y if (y_min - target_value) > (target_value - x_max) else x
                    score[temp] += 1
                    continue
        return self.label[score.argmax()]

    def predict(self, x_test):
        predict = []
        for x in range(len(x_test)):
            print('正在进行第' + str(x) + '个目标的检测')
            predict.append(self.calculate(x_test[x]))
        return predict

    def initialization(self, x_train, y_train):
        self.reshape_dataset(x_train, y_train)
        self.get_mean()
        self.get_sw()
        self.get_w()

    def calculate_accuracy(self, result_label, y_test):
        max = len(result_label)
        correct = 0
        for x in range(len(result_label)):
            if result_label[x] == y_test[x]:
                correct += correct
        result = correct / max
        print("最终的准确率为：%.5f" % result)
