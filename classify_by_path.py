import warnings
import random
from collections import Counter
from urllib.parse import unquote

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle

import preprocess
import pickle
from tools import log_time

warnings.filterwarnings("ignore")


class TfIdfClf:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 1))
        self.clf = LogisticRegression()

    def fit(self, x_train, y_train):
        x_train = self.vectorizer.fit_transform(x_train)
        self.clf.fit(x_train, y_train)

    def predict(self, x_test):
        x_test = self.vectorizer.transform(x_test)
        return self.clf.predict(x_test)


class Classifier:
    def __init__(self, ngram_range=(2, 2), nu=0.008):
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range)
        self.clf = OneClassSVM(kernel='rbf', gamma='auto', verbose=True, max_iter=-1, nu=nu)

    def fit_tf_idf(self, df):
        self.vectorizer.fit(df['Path'])

    def fit_clf(self, df):
        features = self.vectorizer.fit_transform(df['Path']).toarray()
        self.clf.fit(features)

    def predict(self, x_test):
        features = self.vectorizer.transform(x_test['Path']).toarray()
        return self.clf.predict(features)


def split_x_y(df: pd.DataFrame):
    x = df['Path']
    y = df['label']
    return list(x), y


def tokenize(s):
    return s
    # return jieba.lcut(s)
    # r = []
    # n = 2
    # for i in range(len(s) - n + 1):
    #     r.append(s[i:i + n])
    # return s


def read_file(filename, label):
    results = []
    with open(filename, encoding='utf8') as fin:
        for line in fin:
            results.append((tokenize(unquote(line.strip())), label))
    return results


@log_time
def one_class_classifier():
    df = preprocess.basic_data()
    df = df[['Path', 'label']]
    print(df.shape)
    df = df.dropna()
    print(df.shape)
    df = shuffle(df)

    print('Start jieba cut')
    df['Path'] = df['Path'].apply(lambda _p: unquote(eval(_p).decode()))

    # print('Read path good')
    # positive_examples = read_file('data/path/goodqueries.txt', 1)
    # print('Read path bad')
    # negative_examples = read_file('data/path/badqueries.txt', -1)
    # # positive_examples = positive_examples[:int(len(negative_examples) * 2)]
    # all_examples = positive_examples + negative_examples
    # random.shuffle(all_examples)
    # x_train, y_train = zip(*all_examples)
    #
    # print('Start train')
    # clf = TfIdfClf()
    # clf.fit(x_train, y_train)
    # with open('output/vocab.pkl', 'wb') as out:
    #     pickle.dump(clf.vectorizer.vocabulary_, out)

    # df['Path'] = df['Path'].apply(lambda _p: ' '.join(_p))

    # with open('output/vocab.pkl', 'rb') as fin:
    #     vocabulary = pickle.load(fin)
    print('Start Feature Train')
    clfs = []
    for i in range(1, 4):
        clfs.append(Classifier((i, i)))
    for clf in clfs:
        clf.fit_tf_idf(df)
    print('Start Feature Transform')

    positive = df[df['label'] == 0]
    negative = df[df['label'] == 1]

    test_positive = positive.sample(frac=0.2, random_state=0, axis=0)
    train_positive = positive[~positive.index.isin(test_positive.index)]

    print('Start training')
    # train_x, _ = split_x_y(train_positive)
    # train_x, _ = split_x_y(df)
    # print(len(train_x[0]))
    # clf = OneClassSVM(kernel='rbf', gamma='auto', verbose=True, max_iter=-1, nu=0.007)
    # clf = clf.fit(train_x)
    for clf in clfs:
        clf.fit_clf(train_positive)

    print('Start testing positive')
    # x, y = split_x_y(test_positive)
    # x, y = split_x_y(positive)
    # y_pred = clf.predict(x)
    # y = y.apply(lambda x: 1 if x == 0 else -1)
    y_pred = [1] * test_positive.shape[0]
    for clf in clfs:
        _y_pred = clf.predict(test_positive)
        for i in range(_y_pred.shape[0]):
            if _y_pred[i] == -1:
                y_pred[i] = -1

    y = test_positive['label'].apply(lambda x: 1 if x == 0 else -1)
    print(precision_score(y, y_pred, pos_label=1, average='binary'),
          recall_score(y, y_pred, pos_label=1, average='binary'), accuracy_score(y, y_pred))

    print('Start testing negative')
    # x, y = split_x_y(negative)
    # y_pred = clf.predict(x)
    # y = y.apply(lambda x: 1 if x == 0 else -1)
    y_pred = [1] * negative.shape[0]
    for clf in clfs:
        _y_pred = clf.predict(negative)
        for i in range(_y_pred.shape[0]):
            if _y_pred[i] == -1:
                y_pred[i] = -1
    y = negative['label'].apply(lambda x: 1 if x == 0 else -1)
    print(precision_score(y, y_pred, pos_label=-1, average='binary'),
          recall_score(y, y_pred, pos_label=-1, average='binary'), accuracy_score(y, y_pred))

    # print('Start testing positive')
    # x, y = split_x_y(positive)
    # y_pred = clf.predict(x)
    # print(list(y_pred))
    # y = y.apply(lambda x: 1 if x == 0 else -1)
    # print(precision_score(y, y_pred, pos_label=1, average='binary'),
    #       recall_score(y, y_pred, pos_label=1, average='binary'), accuracy_score(y, y_pred))
    #
    # print('Start testing negative')
    # x, y = split_x_y(negative)
    # y_pred = clf.predict(x)
    # print(list(y_pred))
    # y = y.apply(lambda x: 1 if x == 0 else -1)
    # print(precision_score(y, y_pred, pos_label=-1, average='binary'),
    #       recall_score(y, y_pred, pos_label=-1, average='binary'), accuracy_score(y, y_pred))


if __name__ == "__main__":
    one_class_classifier()
