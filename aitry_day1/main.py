import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 加载数据
data = pd.read_csv('data/train.csv')

# 收集模型性能数据
model_names = []
accuracies = []
tprs = []
fprs = []

# 数据预处理
data.fillna({'Age': data['Age'].median()}, inplace=True)
data.fillna({'Embarked': data['Embarked'].mode()[0]}, inplace=True)
data.fillna({'Fare': data['Fare'].median()}, inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

# 将数据分为特征和目标变量
X = data.drop(['Survived', 'Name'], axis=1)
y = data['Survived']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模型
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=200),
    'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
    'SVC': SVC(kernel='rbf')
}

def model_perf(model, y_true, y_pred, name=None):
    """返回模型分类准确率，tpr，fpr"""
    if name is not None:
        print('For model {}: \n'.format(name))
    cm = confusion_matrix(y_true, y_pred)
    if not hasattr(model, 'classes_') and isinstance(model, LinearRegression):
        model.classes_ = [0, 1]
    if hasattr(model, 'classes_'):
        for i in range(len(model.classes_)):
            # 计算真阳性、假阳性、假阴性和真阴性
            tp = cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            fn = cm[i, :].sum() - cm[i, i]
            tn = cm.sum() - tp - fp - fn
            # 计算 TPR, FPR 和 ACC
            tpr = tp / (tp + fn)
            fpr = fp / (tn + fp)
            acc = (tp + tn) / cm.sum()
            print('For class {}: \n TPR is {}; \n FPR is {}; \n ACC is {}. \n'
                  .format(model.classes_[i], tpr, fpr, acc))
    else:
        print('Model {} does not support class-specific metrics.'.format(name))
    return None

def plot_model_performance_line(model_names, accuracies, tprs, fprs,save_path=None):
    """绘制模型性能折线图
    :param model_names: 模型名称列表
    :param accuracies: 准确率列表
    :param tprs: TPR 列表
    :param fprs: FPR 列表
    """
    x = range(len(model_names))

    plt.figure(figsize=(10, 6))

    plt.plot(x, accuracies, marker='o', linestyle='-', color='darkblue', label='Accuracy')
    plt.plot(x, tprs, marker='o', linestyle='--', color='darkblue', label='TPR')
    plt.plot(x, fprs, marker='o', linestyle='-.', color='darkblue', label='FPR')

    plt.xticks(x, model_names, rotation=45)
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_model_performance_bar(model_names, accuracies, tprs, fprs, save_path=None):
    """绘制模型性能柱状图
    :param model_names: 模型名称列表
    :param accuracies: 准确率列表
    :param tprs: TPR 列表
    :param fprs: FPR 列表
    """
    x = range(len(model_names))
    width = 0.2

    plt.figure(figsize=(10, 6))

    plt.bar(x, accuracies, width=width, color='blue', label='Accuracy')
    plt.bar([p + width for p in x], tprs, width=width, color='lightblue', label='TPR')
    plt.bar([p + width*2 for p in x], fprs, width=width, color='darkblue', label='FPR')

    plt.xticks([p + width for p in x], model_names, rotation=45)
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_cm(models, X_test, y_test, save_path=None):
    """画混淆矩阵
    :param models: 模型字典
    :param X_test: 测试集特征
    :param y_test: 测试集标签
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, models.items()):
        predictions = model.predict(X_test)
        if name == 'Linear Regression':
            predictions = np.round(predictions).astype(int)
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(name, fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)  # 设置 x 轴标题字体大小
        ax.set_ylabel(ax.get_ylabel(), fontsize=8)  # 设置 y 轴标题字体大小

    # 添加 color bar
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(disp.im_, cax=cbar_ax)

    fig.subplots_adjust(hspace=0.4, wspace=-0.3)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_cm_ratio(models, X_test, y_test, save_path=None):
    """画混淆矩阵（按占各类型比例）
    :param models: 模型字典
    :param X_test: 测试集特征
    :param y_test: 测试集标签
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, models.items()):
        predictions = model.predict(X_test)
        if name == 'Linear Regression':
            predictions = np.round(predictions).astype(int)
        cm = confusion_matrix(y_test, predictions)
        cm_ratio = np.zeros(cm.shape)
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                cm_ratio[i, j] = cm[i, j] / cm[i].sum()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_ratio)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(name, fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)  # 设置 x 轴标题字体大小
        ax.set_ylabel(ax.get_ylabel(), fontsize=8)  # 设置 y 轴标题字体大小

    # 添加 color bar
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(disp.im_, cax=cbar_ax)

    fig.subplots_adjust(hspace=0.4, wspace=-0.3)
    if save_path:
        plt.savefig(save_path)
    plt.show()

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if name == 'Linear Regression':
        predictions = np.round(predictions)
    accuracy = accuracy_score(y_test, predictions)
    model_names.append(name)
    accuracies.append(accuracy)

    cm = confusion_matrix(y_test, predictions)
    if not hasattr(model, 'classes_') and isinstance(model, LinearRegression):
        model.classes_ = [0, 1]
    if hasattr(model, 'classes_'):
        tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])
        tprs.append(tpr)
        fprs.append(fpr)
    else:
        tprs.append(None)
        fprs.append(None)

# 训练和评估模型
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if name == 'Linear Regression':
        predictions = np.round(predictions)
    accuracy = accuracy_score(y_test, predictions)
    print(f'{name} Accuracy: {accuracy:.2f}')
    survived_names = data.loc[X_test.index[predictions == 1], 'Name']
    # print(f'{name} Predicted Survived Passengers:')
    # print(survived_names.to_list())
    # 调用 model_perf 函数
    model_perf(model, y_test, predictions, name=name)

# 画混淆矩阵并保存
plot_cm(models, X_test, y_test, save_path='cm_plot.png')
plot_cm_ratio(models, X_test, y_test, save_path='cm_ratio_plot.png')

# 绘制模型性能图表
plot_model_performance_line(model_names, accuracies, tprs, fprs,save_path='model_performance(line chart).png')
plot_model_performance_bar(model_names, accuracies, tprs, fprs,save_path='model_performance(bar chart).png')