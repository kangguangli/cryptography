import numpy as np
import pandas as pd
import preprocess
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import MinMaxScaler 
from baggingPU import BaggingClassifierPU
from sklearn.tree import DecisionTreeClassifier

from tools import log_time

def split_x_y(df : pd.DataFrame) :

    x = df[[col for col in df.columns if col != 'label']]
    y = df['label']

    return x, y

@log_time
def bagging_classifier():
    df = preprocess.basic_data()
    # 此时df内的数据包含了csv的所有列，具体使用了那些csv文件见函数内部，如果想使用更多数据可以在此函数基础上修改
    # 这里的df“好像”可以直接喂给一些对数据格式要求不严格的分类器，比如lightgbm，如果我没记错的话，也可能我记错了

    df = preprocess.basic_process(df)
    # 在此函数中对特征进行了一些裁剪，细节见函数内部，代码比较简单，不详细介绍了
    # 这列的df可以直接喂给分类器
    # 如果想对特征做更精细的处理，可以在此函数基础上进行修改

    df = preprocess.vica_further_process(df)

    # 训练分类器，分别计算正负例的精度和召回率

    sclaer = MinMaxScaler()

    postive = df[df['label'] == 0]
    negtive = df[df['label'] != 0]
    # print(postive)
    # print(negtive)

    test_postive = postive.sample(frac=0.2, random_state=0, axis=0)
    train_postive = postive[~postive.index.isin(test_postive.index)]

    test_negtive = negtive.sample(frac=0.2, random_state=0, axis=0)
    train_negtive = negtive[~negtive.index.isin(test_negtive.index)]

    P = train_postive.sample(frac=0.5, random_state=0, axis=0)
    U_part1 = train_postive[~train_postive.index.isin(P.index)]

    P_x, _ = split_x_y(P)
    U_part1_x, _ = split_x_y(U_part1)
    train_negtive_x, _ = split_x_y(train_negtive)

    U_x = np.r_[U_part1_x.values, train_negtive_x.values]
    train_x = sclaer.fit_transform(np.r_[P_x.values, U_x])
    train_y = np.r_[np.ones(len(P_x.values)), np.zeros(len(U_x))]
    print(len(P_x.values))
    print(len(U_x))

    bc = BaggingClassifierPU(
        DecisionTreeClassifier(), 
        n_estimators = 300,  # 1000 trees as usual
        max_samples = len(P_x.values), # Balance the positives and unlabeled in each bag
        n_jobs = -1           # Use all cores
        )
    bc = bc.fit(train_x, train_y)

    px, py = split_x_y(test_postive)
    nx, ny = split_x_y(test_negtive)
    test_x = np.r_[px.values, nx.values]
    test_y = np.r_[np.ones(len(px.values)), np.zeros(len(nx.values))]
    y_pred = bc.predict(sclaer.fit_transform(test_x))
    print(y_pred)

    p1 = precision_score(test_y, y_pred, pos_label = 1, average = 'binary')
    r1 = recall_score(test_y, y_pred, pos_label = 1, average = 'binary')
    f1 = f1_score(test_y, y_pred, pos_label = 1, average = 'binary')

    fpr, tpr, _ = roc_curve(test_y, y_pred, pos_label = 1)
    auc1 = auc(fpr, tpr)

    print(p1, r1, f1, auc1)

    p0 = precision_score(test_y, y_pred, pos_label = 0, average = 'binary')
    r0 = recall_score(test_y, y_pred, pos_label = 0, average = 'binary')
    f0 = f1_score(test_y, y_pred, pos_label = 0, average = 'binary')

    fpr, tpr, _ = roc_curve(test_y, y_pred, pos_label = 0)
    auc0 = auc(fpr, tpr)

    print(p0, r0, f0, auc0)


if __name__ == "__main__":
    
    bagging_classifier()