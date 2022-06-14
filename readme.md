# 主成分分析与 Fisher 线性判别

由于单一图片的像素点过多（19800个），因而自定义 PCA 在进行特征向量提取时需要耗费大量时间，因而 主成分分析算法采用 sklean 包中自带的 PCA 降维方法。

题目中要求使用 Fisher 线性判别，因而自定义了一个 Fisher 类，通过其生成的对象可以进行 Fisher 判别。

题中要求使用最近邻分类算法，但不是主要内容，因而采用 sklean 保重自带的 k 近邻算法。

main.py 的主函数中分别调用了 3 个函数 task1() task2() task3()，分别对应了实验二中的 基本要求、中级要求和高级要求。

同时，代码已备份至[Github](https://github.com/Mundanity-fc/PCA_with_Fisher_on_AR_Face_Database)。

测试环境为 Python 3.9，所需第三方包已在 requirements.txt 中给出，执行`pip install -r requirements.txt` 进行安装

使用到的两个自定义类 ImgProcess类 和 Fisher类，其说明在下面给出。

### ImgProcess类

ImgProcess 类包含了对图片数据的加载、处理以及标签的提取。

- load_img_data()——实现了对指定路径下的图片的加载，并进行了灰度化处理，最后的结果作为数据集，同时根据文件名对其进行标签划分
- get_split_index()——随机指定每个人的 7 张图片作为测试集
- dataset_split()——对数据集进行划分，以 7 : 3 的比例划分为训练集和测试集
- run()——运行上述所有函数，并返回训练集的数据与标签和测试集的数据与标签

### Fisher类

Fisher 类实现了 Fisher 线性判别的有关内容。由于 Fisher 判别器为二分类判别器，因而在多分类时需要额外操作。对某一特定数据进行分类时，需要遍历所有的分类结果，并返回被划分次数最多的类别。

- reshape_dataset()——根据标签内容，对同类的测试数据集进行聚合，相同类别的数据集的 ndarry 保存至同一个 list 当中。
- get_mean()——对每一类的数据进行均值计算
- get_sw()——根据计算所得的均值再次计算两类之间的类内散度和总类内散度，由于 Fisher 判别是二分类操作，因而共有 $C_{100}^{2}$ 个有用数据
- get_w()——根据计算所得的散度再次计算两类之间的映射向量
- get_range()——对指定的数据集和指定的映射向量计算对应数据集在一维空间中的坐标范围上下界
- calculate()——对指定的测试数据进行标签预测
- predict()——对指定多个测试集进行标签预测
- initialization()——根据已有参数，初始化 Fisher 判别器，计算所需参数
- calculate_accuracy()——根据测试标签集，计算分类的准确度