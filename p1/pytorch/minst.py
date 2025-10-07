from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np
#
# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 学习曲线绘制函数
# def plot_learning_curve(estimator, X, y):
#    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5)
#    train_mean = np.mean(train_scores, axis=1)
#    test_mean = np.mean(test_scores, axis=1)
#    plt.plot(train_sizes, train_mean, label="Training Score")
#    plt.plot(train_sizes, test_mean, label="Validation Score")
#    plt.xlabel("Training Size")
#    plt.ylabel("Accuracy")
#    plt.legend()
#    plt.title("Learning Curve")
#    plt.show()
# # 随机森林模型
# rf = RandomForestClassifier(max_depth=10, n_estimators=50)
# plot_learning_curve(rf, X_train[:5000], y_train[:5000]) # 使用部分数据加速计算
#
# # 验证曲线绘制函数
# param_range = [10, 20, 40, 80, 160, 250]
# train_scores, test_scores = validation_curve(
#    RandomForestClassifier(), X_train[:5000], y_train[:5000],
#    param_name="n_estimators", param_range=param_range, cv=3)
# train_mean = np.mean(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# plt.plot(param_range, train_mean, label="Training Score")
# plt.plot(param_range, test_mean, label="Validation Score")
# plt.xlabel("Number of Estimators")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.title("Validation Curve for n_estimators")
# plt.show()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# SVM模型训练与参数调整
svm_model = SVC(kernel='rbf', C=10, gamma=0.01, verbose=True)
svm_model.fit(X_train[:5000], y_train[:5000]) # 使用部分数据加速计算
# 模型评估
y_pred = svm_model.predict(X_test[:1000])
print("SVM Accuracy:", accuracy_score(y_test[:1000], y_pred))
