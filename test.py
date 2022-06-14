from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from customPCA import customPCA
from ImgProcess import ImgProcess
from sklearn.decomposition import PCA
from Fisher import Fisher


def task1():
    """
    基本要求——将AR人脸数据集划分成训练集和测试集，实现主成份分析算法，将人脸图像压缩成低维特征向量，并结合最近邻分类算法计算分类精度，分析实验结果。
    :return: 无返回值
    """""
    imgProcess = ImgProcess('./dataset/')
    x_train, y_train, x_test, y_test = imgProcess.run()
    pca = PCA(n_components=50, whiten=True, random_state=0).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train_pca, y_train)
    prediction = knn.predict(x_test_pca)
    print(accuracy_score(y_test, prediction))


def task2():
    """
    中级要求——展示用主成分分析将模式样本压缩成不同特征维度时，模式分类准确率。

    :return: 无返回值
    """""
    imgProcess = ImgProcess('./dataset/')
    x_train, y_train, x_test, y_test = imgProcess.run()
    pca1 = PCA(n_components=50, whiten=True, random_state=0).fit(x_train)
    x_train_pca1 = pca1.transform(x_train)
    x_test_pca1 = pca1.transform(x_test)
    pca2 = PCA(n_components=250, whiten=True, random_state=0).fit(x_train)
    x_train_pca2 = pca2.transform(x_train)
    x_test_pca2 = pca2.transform(x_test)
    pca3 = PCA(n_components=500, whiten=True, random_state=0).fit(x_train)
    x_train_pca3 = pca3.transform(x_train)
    x_test_pca3 = pca3.transform(x_test)
    pca4 = PCA(n_components=1000, whiten=True, random_state=0).fit(x_train)
    x_train_pca4 = pca4.transform(x_train)
    x_test_pca4 = pca4.transform(x_test)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train_pca1, y_train)
    prediction = knn.predict(x_test_pca1)
    print('保留50个特征值时准确率为：%.5f' % accuracy_score(y_test, prediction))
    knn.fit(x_train_pca2, y_train)
    prediction = knn.predict(x_test_pca2)
    print('保留250个特征值时准确率为：%.5f' % accuracy_score(y_test, prediction))
    knn.fit(x_train_pca3, y_train)
    prediction = knn.predict(x_test_pca3)
    print('保留500个特征值时准确率为：%.5f' % accuracy_score(y_test, prediction))
    knn.fit(x_train_pca4, y_train)
    prediction = knn.predict(x_test_pca4)
    print('保留1000个特征值时准确率为：%.5f' % accuracy_score(y_test, prediction))


def task3():
    """
    高级要求——对人脸数据集先进行主成分分析，在此基础上再做Fisher判别分析，计算分类精度，分析实验结果。

    :return: 无返回值
    """""
    imgProcess = ImgProcess('./dataset/')
    x_train, y_train, x_test, y_test = imgProcess.run()
    FDA = Fisher()
    pca = PCA(n_components=50, whiten=True, random_state=0).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    FDA.initialization(x_train_pca, y_train)
    result = FDA.predict(x_test_pca)
    FDA.calculate_accuracy(result, y_test)


if __name__ == '__main__':
    task3()
