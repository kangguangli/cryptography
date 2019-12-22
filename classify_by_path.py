import warnings
import random
from collections import Counter
from urllib.parse import unquote
import PLU
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

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
    def __init__(self, ngram_range=(2, 2), nu=0.0055, analyzer='char_wb', tokenizer=None):
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, tokenizer=tokenizer)
        # self.clf = EllipticEnvelope(contamination=0.05)
        # self.clf = IsolationForest(contamination=0.03)
        self.clf = OneClassSVM(kernel='rbf', gamma='auto', verbose=True, max_iter=-1, nu=nu)

    def fit_tf_idf(self, df):
        self.vectorizer.fit(df['Path'])

    def fit_clf(self, df):
        features = self.vectorizer.fit_transform(df['Path']).toarray()
        self.clf.fit(features)

    def transform_tf_idf(self, x_test):
        return self.vectorizer.transform(x_test['Path']).toarray()

    def predict(self, x_test):
        features = self.vectorizer.transform(x_test['Path']).toarray()
        return self.clf.predict(features)


def split_x_y(df: pd.DataFrame):
    x = df['Path']
    y = df['label']
    return list(x), y


def tokenize(s):
    # return s
    # return ' '.join(jieba.lcut(s))
    return jieba.lcut(s)
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


def get_most_frq_words(vectorizer, df):
    f = vectorizer.transform(df['Path']).toarray()
    vocab = vectorizer.get_feature_names()
    chs = list(map(lambda x: vocab[x[0]], sorted(enumerate(np.sum(f, 0)), key=lambda x: x[1], reverse=True)[:50]))
    print(' '.join(chs))
    return chs
    # print(sorted(vectorizer.vocabulary_.items(), reverse=True))
    # print(' '.join(map(lambda x: x[0], sorted(vectorizer.vocabulary_.items(), reverse=True))))


@log_time
def one_class_classifier():
    df = preprocess.basic_data()
    print('>>>>>>>>>>>', len(df['Path'].unique()))
    df['Path'] = df['Path'].fillna(df['Raw_load'])
    print('>>>>>>>>>>>', len(df['Path'].unique()))
    df = df[['Path', 'label']]

    print(df.shape)
    df = df.dropna()
    print(df.shape)
    df = shuffle(df)

    print('Start jieba cut')
    df['Path'] = df['Path'].apply(lambda _p: unquote(eval(_p).decode()))

    positive = df[df['label'] == 0]
    negative = df[df['label'] == 1]

    # vectorizer = CountVectorizer(analyzer='char')
    # vectorizer.fit(df['Path'])
    # print('[正例top 50]\t', end='')
    # pchs = set(get_most_frq_words(vectorizer, positive))
    # print('[负例top 50]\t', end='')
    # nchs = set(get_most_frq_words(vectorizer, negative))
    # print('[正例独有]\t', ' '.join(pchs - nchs))
    # print('[负例独有]\t', ' '.join(nchs - pchs))
    # return

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
    for i in range(1, 2):
        # clfs.append(Classifier((i, i), 0.005))
        # clfs.append(Classifier((i, i), 0.003))
        clfs.append(Classifier((i, i), 0.0055))
        # clfs.append(Classifier((i, i), 0.0058))
        # clfs.append(Classifier((i, i), 0.006))
    # clfs.append(Classifier(ngram_range=(1, 1), analyzer='word', tokenizer=tokenize))
    for clf in clfs:
        clf.fit_tf_idf(df)
    print('Start Feature Transform')

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
    not_always_true = -1
    print('Start testing positive')
    # x, y = split_x_y(test_positive)
    # x, y = split_x_y(positive)
    # y_pred = clf.predict(x)
    # y = y.apply(lambda x: 1 if x == 0 else -1)
    y_pred = [not_always_true] * test_positive.shape[0]
    for clf in clfs:
        _y_pred = clf.predict(test_positive)
        for i in range(_y_pred.shape[0]):
            if _y_pred[i] == -not_always_true:
                y_pred[i] = -not_always_true

    y = test_positive['label'].apply(lambda x: 1 if x == 0 else -1)
    print(precision_score(y, y_pred, pos_label=1, average='binary'),
          recall_score(y, y_pred, pos_label=1, average='binary'), accuracy_score(y, y_pred))

    print('Start testing negative')
    # x, y = split_x_y(negative)
    # y_pred = clf.predict(x)
    # y = y.apply(lambda x: 1 if x == 0 else -1)
    y_pred = [not_always_true] * negative.shape[0]
    for clf in clfs:
        _y_pred = clf.predict(negative)
        for i in range(_y_pred.shape[0]):
            if _y_pred[i] == -not_always_true:
                y_pred[i] = -not_always_true
    y = negative['label'].apply(lambda x: 1 if x == 0 else -1)
    print(len(positive))
    print(len(negative))
    print(precision_score(y, y_pred, pos_label=-1, average='binary'),
          recall_score(y, y_pred, pos_label=-1, average='binary'), accuracy_score(y, y_pred))

    print('Start testing all')
    # x, y = split_x_y(negative)
    # y_pred = clf.predict(x)
    # y = y.apply(lambda x: 1 if x == 0 else -1)
    y_pred = [not_always_true] * df.shape[0]
    for clf in clfs:
        _y_pred = clf.predict(df)
        for i in range(_y_pred.shape[0]):
            if _y_pred[i] == -not_always_true:
                y_pred[i] = -not_always_true
    y = df['label'].apply(lambda x: 1 if x == 0 else -1)
    print(len(positive))
    print(len(negative))
    print(precision_score(y, y_pred, pos_label=-1, average='binary'),
          recall_score(y, y_pred, pos_label=-1, average='binary'), accuracy_score(y, y_pred))

    print(precision_score(y, y_pred, pos_label=1, average='binary'),
          recall_score(y, y_pred, pos_label=1, average='binary'), accuracy_score(y, y_pred))

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
    # pu()
