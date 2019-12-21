import pandas as pd
import preprocess
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from tools import log_time


def split_x_y(df: pd.DataFrame):
    x = df[[col for col in df.columns if col != 'label']]
    y = df['label']

    return x, y


@log_time
def one_class_classifier():
    df = preprocess.basic_data()
    df = preprocess.basic_process(df)

    sclaer = MinMaxScaler()

    postive = df[df['label'] == 0]
    negtive = df[df['label'] != 0]

    test_postive = postive.sample(frac=0.2, random_state=0, axis=0)
    train_postive = postive[~postive.index.isin(test_postive.index)]

    # test_negtive = negtive.sample(frac = 0.2, random_state = 0, axis = 0)
    # train_negtive = negtive[~negtive.index.isin(test_negtive.index)]
    # unlabeled = train_postive.sample(frac = 0.2, random_state = 0, axis = 0)
    # test = pd.concat([test_postive, negtive], ignore_index = True)

    train_x, _ = split_x_y(train_postive)
    clf = OneClassSVM(kernel='rbf', gamma='auto', verbose=True, max_iter=10000)
    clf = clf.fit(sclaer.fit_transform(train_x))

    x, y = split_x_y(test_postive)
    y_pred = clf.predict(sclaer.fit_transform(x))
    y = y.apply(lambda x: 1 if x == 0 else -1)
    print(precision_score(y, y_pred, pos_label=1, average='binary'),
          recall_score(y, y_pred, pos_label=1, average='binary'), accuracy_score(y, y_pred))

    x, y = split_x_y(negtive)
    y_pred = clf.predict(sclaer.fit_transform(x))
    y = y.apply(lambda x: 1 if x == 0 else -1)
    print(precision_score(y, y_pred, pos_label=-1, average='binary'),
          recall_score(y, y_pred, pos_label=-1, average='binary'), accuracy_score(y, y_pred))


if __name__ == "__main__":
    one_class_classifier()
