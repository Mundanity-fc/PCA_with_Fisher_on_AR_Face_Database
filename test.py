from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from customPCA import customPCA
from ImgProcess import ImgProcess
from sklearn.decomposition import PCA
from Fisher import Fisher


def task1():
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
    imgProcess = ImgProcess('./dataset/')
    x_train, y_train, x_test, y_test = imgProcess.run()
    pca1 = PCA(n_components=50, whiten=True, random_state=0).fit(x_train)
    x_train_pca1 = pca1.transform(x_train)
    x_test_pca1 = pca1.transform(x_test)
    pca2 = PCA(n_components=500, whiten=True, random_state=0).fit(x_train)
    x_train_pca2 = pca1.transform(x_train)
    x_test_pca2 = pca1.transform(x_test)
    pca3 = PCA(n_components=5000, whiten=True, random_state=0).fit(x_train)
    x_train_pca3 = pca1.transform(x_train)
    x_test_pca3 = pca1.transform(x_test)
    pca4 = PCA(n_components=10000, whiten=True, random_state=0).fit(x_train)
    x_train_pca4 = pca1.transform(x_train)
    x_test_pca4 = pca1.transform(x_test)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train_pca1, y_train)
    prediction = knn.predict(x_test_pca1)
    print('保留50个特征值时准确率为：%.5f' % accuracy_score(y_test, prediction))
    knn.fit(x_train_pca2, y_train)
    prediction = knn.predict(x_test_pca2)
    print('保留500个特征值时准确率为：%.5f' % accuracy_score(y_test, prediction))
    knn.fit(x_train_pca3, y_train)
    prediction = knn.predict(x_test_pca3)
    print('保留5000个特征值时准确率为：%.5f' % accuracy_score(y_test, prediction))
    knn.fit(x_train_pca4, y_train)
    prediction = knn.predict(x_test_pca4)
    print('保留10000个特征值时准确率为：%.5f' % accuracy_score(y_test, prediction))


def task3():
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
