from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# 加载Iris数据集
iris = load_iris()
X = iris.data
y = iris.target

# 选取前两类样本并取出sepal length和sepal width作为特征
X = X[:100, :2]
y = y[:100]


# 划分训练集和测试集
X_train = np.vstack((X[:40], X[50:90]))
y_train = np.hstack((y[:40], y[50:90]))
X_test = np.vstack((X[40:50], X[90:100]))
y_test = np.hstack((y[40:50], y[90:100]))

# 绘制原始训练数据分布的散点图
plt.scatter(X[:40, 0], X[:40, 1], label='Class 0', c='b', marker='o')
plt.scatter(X[50:90, 0], X[50:90, 1], label='Class 1', c='r', marker='x')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

class Perceptron:
    def __init__(self, lr):
        self.lr = lr #学习率
        self.w = np.ones(X_train.shape[1] + 1)  # 权重向量初始化为1
        self.w[0] = 0  #偏置初始化为0
        self.errors = []

    def predict(self, x):                          #如果线性组合的结果大于等于0，则返回类别1，否则返回类别0
        return np.where(np.dot(x, self.w[1:]) + self.w[0] >= 0, 1, 0)

    def fit(self, X, y):          #迭代的停止条件为“直到训练集内没有误分类样本为止”
        converged = False
        while not converged:
            error = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                error += int(update != 0.0)
            self.errors.append(error)
            if error == 0:
                converged = True
        return self

# 训练感知机模型
model = Perceptron(lr=0.1)
model.fit(X_train, y_train)

# 预测测试集
predictions = model.predict(X_test)

# 计算准确率等评估指标
accuracy = np.mean(predictions == y_test)
print('epoch:', len(model.errors))
print('learning rate:', model.lr)
print("Accuracy:", accuracy)

# 绘制分类边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
#训练集图像
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:40, 0], X[:40, 1], label='Class 0', c='b', marker='o')
plt.scatter(X[50:90, 0], X[50:90, 1], label='Class 1', c='r', marker='x')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
#测试集图像
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[40:50, 0], X[40:50, 1], label='Class 0', c='b', marker='o')
plt.scatter(X[90:100, 0], X[90:100, 1], label='Class 1', c='r', marker='x')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()