import numpy


class customPCA:

    def __init__(self, x_train, k):
        self.x_data = x_train
        self.k = k

    def run(self):
        print('\n正在进行PCA降维操作……')
        features = len(self.x_data[0])
        mean = numpy.array([numpy.mean(self.x_data[:, i]) for i in range(features)])
        norm = self.x_data - mean
        scatter_matrix = numpy.dot(numpy.transpose(norm), norm)
        eig_val, eig_vec = numpy.linalg.eig(scatter_matrix)
        eig_pairs = [(numpy.abs(eig_val[i]), eig_vec[:, i]) for i in range(features)]
        eig_pairs.sort(reverse=True)
        feature = numpy.array([ele[1] for ele in eig_pairs[:self.k]])
        result = numpy.dot(norm, numpy.transpose(feature))
        print('\n完成了PCA降维操作')
        return result
