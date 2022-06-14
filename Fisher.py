import numpy


class Fisher:
    def __init__(self):
        self.w = []
        self.sw = []
        self.data = []
        self.dataMean = []

    def reshape_dataset(self, x_train, y_train):
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
                tempX = x
        self.data.append(numpy.array(tempList))
        self.data.pop(0)
        self.data = numpy.array(self.data)

    def get_mean(self):
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

    def get_sw(self):
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

    def calculate(self, x_train, y_train):
        self.reshape_dataset(x_train, y_train)
        self.get_mean()
        # self.get_sw()
        # x = self.sw[0][1]
        # y = numpy.linalg.inv(x)
        # z = self.dataMean[0] - self.dataMean[1]
        # r = numpy.dot(y, z)
        print()
