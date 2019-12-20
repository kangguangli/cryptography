from sklearn import naive_bayes
from sklearn.feature_extraction import text
from sklearn import neighbors
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
# two stage
# use

# stage1 = ['Bayes', 'Rocchio', 'Spy']
# stage2 = ['SVM', 'Biased SVM', 'Iterative SVM', 'Logistic']
#     y_total_pred = np.zeros(len(train_u.values)+len(a.values))
#     for i in stage1:
# P = train_postive.values
# U = train_unlabel.values
# plu = PLU.PLU(sclaer.fit_transform(P), sclaer.fit_transform(U))
# RN = plu.stage_one(i)
# y_pred = plu.stage_two(RN,stage2)

class PLU:
    def __init__(self, P, U):
        self.P = P  # P集由数据集中的正样本构成
        self.U = U  # U集由数据集中未标注的样本构成

    def minus(self, a, b):
        l = []
        for i in range(len(a)):
            if a[i] in b:
                l.append(1)
            else:
                l.append(0)
        return a[np.array(l) == 0]

    def add(self, a, b):
        return np.r_[self.minus(a, b), b]

    def get_RN_Bayes(self):
        pos = self.P
        pos_label = np.ones(len(pos))
        un = self.U
        un_label = np.zeros(len(un))

        X_train = np.r_[pos, un]
        y_train = np.r_[pos_label, un_label]
        X_test = self.U

        clf = naive_bayes.MultinomialNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        RN = self.U[y_pred == 0] 
        return RN
    
    def get_RN_Rocchio(self):
        pos = self.P
        pos_label = np.ones(len(pos))
        un = self.U
        un_label = np.zeros(len(un))

        X_train = np.r_[pos, un]
        y_train = np.r_[pos_label, un_label]
        # vec = text.CountVectorizer()
        # X_train = vec.fit_transform(X_train)
        tfidf_transformer = text.TfidfTransformer()
        X_train = tfidf_transformer.fit_transform(X_train)
        X_test = self.U
        # vec = text.CountVectorizer()
        # X_test = vec.fit_transform(X_test)
        tfidf_transformer = text.TfidfTransformer()
        X_test = tfidf_transformer.fit_transform(X_test)
    
        clf = neighbors.NearestCentroid()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        RN = self.U[y_pred == 0]
        return RN
        
    def get_RN_Spy(self):
        p_indices = np.random.RandomState(1).permutation(len(self.P))
        spy_num = int(len(self.P) * 0.15)
        spy = self.P[p_indices[:spy_num]]

        pos = self.P[p_indices[spy_num:]]
        pos_label = np.ones(len(pos))
        un = np.r_[self.U, spy]
        un_label = np.zeros(len(un))

        X_train = np.r_[pos, un]
        y_train = np.r_[pos_label, un_label]

        clf = naive_bayes.MultinomialNB()
        clf.fit(X_train, y_train)
        y_prob_un = clf.predict_proba(self.U)[:, -1]
        y_prob_spy = clf.predict_proba(spy)[:, -1]
        th = np.min(y_prob_spy)
        RN = self.U[y_prob_un < th] 
        return RN

    def stage_one(self, type):
        RN = np.array([])
        if type == 'Bayes':
            RN = self.get_RN_Bayes()
        elif type == 'Rocchio':
            RN = self.get_RN_Rocchio()
        elif type == 'Spy':
            RN = self.get_RN_Spy()
        return RN

    def get_pred_SVM(self, RN):
        RN_label = np.zeros(len(RN))
        pos = self.P
        pos_label = np.ones(len(pos))

        X_train = np.r_[pos, RN]
        y_train = np.r_[pos_label, RN_label]
        X_test = self.U

        clf = svm.SVC(kernel = 'rbf', gamma = 'auto', class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def get_pred_Biased_SVM(self, RN):
        RN_label = np.zeros(len(RN))
        pos = self.P
        pos_label = np.ones(len(pos))

        X_train = np.r_[pos, RN]
        y_train = np.r_[pos_label, RN_label]
        X_test = self.U
        cost_fn = 1
        cost_fp = 3
        weight = [cost_fn if i else cost_fp for i in y_train]
        clf = svm.SVC(kernel = 'rbf', gamma = 'auto')
        clf.fit(X_train, y_train, sample_weight=weight) 
        y_pred = clf.predict(X_test)
        return y_pred

    def get_pred_Iterative_SVM(self, RN):
        RN_label = np.zeros(len(RN))
        pos = self.P
        pos_label = np.ones(len(pos))

        X_train = np.r_[pos, RN]
        y_train = np.r_[pos_label, RN_label]
        X_test = self.U
        Q = self.minus(X_test, RN)
        NRN = RN
        while True:
            clf = svm.SVC()
            clf.fit(X_train, y_train)
            if len(Q) == 0: 
                break
            y_pred = clf.predict(Q)
            W = Q[y_pred == 0]
            if len(W) == 0: 
                break
            else:
                Q = self.minus(Q, W)
                NRN = self.add(Q, NRN)
                RN_label = np.zeros(len(NRN))
                X_train = np.r_[pos, NRN]
                y_train = np.r_[pos_label, RN_label]
        y_pred = clf.predict(X_test)
        print(len(X_test))
        return y_pred

    def get_pred_Logistic(self, RN):
        RN_label = np.zeros(len(RN))
        pos = self.P
        pos_label = np.ones(len(pos))

        X_train = np.r_[pos, RN]
        y_train = np.r_[pos_label, RN_label]
        X_test = self.U

        clf = LogisticRegression(class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def stage_two(self, RN, type):
        pred =  np.array([])
        if type == 'SVM':
            pred = self.get_pred_SVM(RN)
        elif type == 'Biased SVM':
            pred = self.get_pred_Biased_SVM(RN)
        elif type == 'Iterative SVM':
            pred = self.get_pred_Iterative_SVM(RN)
        elif type == 'Logistic':
            pred = self.get_pred_Logistic(RN)
        return pred