import pandas as pd
import preprocess
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler

from tools import log_time


def split_x_y(df: pd.DataFrame):
    x = df[[col for col in df.columns if col != 'label']]
    y = df['label']

    return x, y


@log_time
def one_class_classifier():
    df = preprocess.basic_data()
    # 此时df内的数据包含了csv的所有列，具体使用了那些csv文件见函数内部，如果想使用更多数据可以在此函数基础上修改
    # 这里的df“好像”可以直接喂给一些对数据格式要求不严格的分类器，比如lightgbm，如果我没记错的话，也可能我记错了

    df = preprocess.basic_process(df, True)
    # 在此函数中对特征进行了一些裁剪，细节见函数内部，代码比较简单，不详细介绍了
    # 这列的df可以直接喂给分类器
    # 如果想对特征做更精细的处理，可以在此函数基础上进行修改

    df = preprocess.vica_further_process(df)

    # 训练分类器，分别计算正负例的精度和召回率

    sclaer = MinMaxScaler()

    postive = df[df['label'] == 0]
    negtive = df[df['label'] != 0]

    test_postive = postive.sample(frac=0.2, random_state=0, axis=0)
    train_postive = postive[~postive.index.isin(test_postive.index)]

    train_x, _ = split_x_y(train_postive)

    # OneClassSVM用来做离群检验，可以检测异常数据，适用于只知道正例的情况，是一种无监督方法
    # 下面的参数中相对比较重要的是nu，代表输入数据中异常数据比例的上界
    # 此时输入的全为正例，因此将其设为一个比较小的值
    # clf = OneClassSVM(kernel='rbf', gamma='auto', max_iter=-1, random_state=0, nu=1e-6)
    clf = OneClassSVM(kernel='rbf', gamma='auto', max_iter=-1, nu=1e-6)
    clf = clf.fit(sclaer.fit_transform(train_x))

    test = pd.concat([test_postive, negtive])
    x, y = split_x_y(test)

    y_pred = clf.predict(sclaer.transform(x))
    y = y.apply(lambda x: 1 if x == 0 else -1)

    p1 = precision_score(y, y_pred, pos_label=1, average='binary')
    r1 = recall_score(y, y_pred, pos_label=1, average='binary')
    f1 = f1_score(y, y_pred, pos_label=1, average='binary')

    fpr, tpr, _ = roc_curve(y, y_pred, pos_label=1)
    auc1 = auc(fpr, tpr)

    print(p1, r1, f1, auc1)

    p0 = precision_score(y, y_pred, pos_label=-1, average='binary')
    r0 = recall_score(y, y_pred, pos_label=-1, average='binary')
    f0 = f1_score(y, y_pred, pos_label=-1, average='binary')

    fpr, tpr, _ = roc_curve(y, y_pred, pos_label=-1)
    auc0 = auc(fpr, tpr)

    print(p0, r0, f0, auc0)


if __name__ == "__main__":
    one_class_classifier()
