import pandas as pd
import preprocess
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import PLU
import numpy as np
from tools import log_time
import random

def split_x_y(df : pd.DataFrame) :

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

    # df = preprocess.n_gram_path_process(df)
    #df = df.drop('Path', axis=1)

    # 训练分类器，分别计算正负例的精度和召回率

    sclaer = MinMaxScaler()


    postive = df[df['label'] == 0]
    negtive = df[df['label'] != 0]

    test_postive = postive.sample(frac = 0.2, random_state = 0, axis = 0)
    train_postive = postive[~postive.index.isin(test_postive.index)]

    train_x, _ = split_x_y(train_postive)
    test_x, _ = split_x_y(test_postive)
    train_u, _ = split_x_y(negtive)

    stage1 = ['Bayes', 'Rocchio', 'Spy']
    stage2 = ['SVM', 'Biased SVM', 'Iterative SVM', 'Logistic']
    P = train_x.values
    U = np.r_[test_x.values, train_u.values]
    y_test = np.r_[np.ones(len(test_x.values)), np.zeros(len(train_u.values))]
    # all_data = list(zip(U, y_test))
    # random.shuffle(all_data)
    # U, y_test = zip(*all_data)
    sclaer.fit(P)
    plu = PLU.PLU(sclaer.transform(P), sclaer.transform(U))
    # plu = PLU.PLU(P, U)

    RN = plu.stage_one(stage1[0])
    print(len(RN))
    y_pred = plu.stage_two(RN,stage2[3])
    print(np.sum(y_pred), len(y_pred)-np.sum(y_pred))
    print(accuracy_score(y_test, y_pred))
    print(precision_score(y_test, y_pred, pos_label=0))
    print(recall_score(y_test, y_pred, pos_label=0))
    print(f1_score(y_test, y_pred, pos_label=0))
    print(precision_score(y_test, y_pred, pos_label=1))
    print(recall_score(y_test, y_pred, pos_label=1))
    print(f1_score(y_test, y_pred, pos_label=1))
    print(roc_auc_score(y_test, y_pred))
    # OneClassSVM用来做离群检验，可以检测异常数据，适用于只知道正例的情况，是一种无监督方法
    # 下面的参数中相对比较重要的是nu，代表输入数据中异常数据比例的上界
    # 此时输入的全为正例，因此将其设为一个比较小的值
    # clf = OneClassSVM(kernel = 'rbf', gamma = 'auto', verbose = True, max_iter = -1, random_state = 0, nu = 1e-6)
    # clf = clf.fit(sclaer.fit_transform(train_x))

    # x, y = split_x_y(test_postive)
    # y_pred = clf.predict(sclaer.fit_transform(x))
    # y = y.apply(lambda x : 1 if x == 0 else -1)
    # print(precision_score(y, y_pred, pos_label = 1, average = 'binary'), recall_score(y, y_pred, pos_label = 1, average = 'binary'), accuracy_score(y, y_pred))

    # x, y = split_x_y(negtive)
    # y_pred = clf.predict(sclaer.fit_transform(x))
    # y = y.apply(lambda x : 1 if x == 0 else -1)
    # print(precision_score(y, y_pred, pos_label = -1, average = 'binary'), recall_score(y, y_pred, pos_label = -1, average = 'binary'), accuracy_score(y, y_pred))


if __name__ == "__main__":

    one_class_classifier()
