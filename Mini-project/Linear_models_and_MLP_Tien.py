# project by: Tien Li Shen & Suki Zhu
# Date: 11/11/2020

from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
import pandas as pd

import numpy as np

def load_files():
    x_train = pd.read_csv("train_features.csv")#, nrows = 1500)
    y_train_scored = pd.read_csv("train_targets_scored.csv")#, nrows  = 1500)
    y_train_nonscored = pd.read_csv("train_targets_nonscored.csv")#, nrows  = 1500)
    x_test = pd.read_csv("test_features.csv")
    return x_train, y_train_scored, y_train_nonscored, x_test

def OneVsRest_SVC(x_train, x_test, y_train, y_test):
    # model = RidgeClassifier(alpha=0)
    model = SVC(kernel = "linear")
    clf = OneVsRestClassifier(model).fit(x_train, y_train)
    print("fit success")

    y_pred_prob = clf.predict_proba(x_test)

    y_pred = clf.predict(x_test)

    print("y_pred:\n", y_pred)

    loss = log_loss(y_test, y_pred_prob)
    print("log_loss: ", loss)

    f1 = f1_score(y_test, y_pred, average="micro")
    print("f1 score: ", f1)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: ", accuracy)

    return [f1, accuracy]


def OneVsRest_Ridge(x_train, x_test, y_train, y_test):
    model = RidgeClassifier(alpha=0)
    # model = SVC(kernel="linear", probability=True)
    clf = OneVsRestClassifier(model).fit(x_train, y_train)
    print("fit success")

    y_pred_prob = clf.predict_proba(x_test)

    y_pred = clf.predict(x_test)

    print("y_pred:\n", y_pred)

    loss = log_loss(y_test, y_pred_prob)
    print("log_loss: ", loss)

    f1 = f1_score(y_test, y_pred, average="micro")
    print("f1 score: ", f1)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: ", accuracy)

    return [f1, accuracy]

def randomF(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(max_depth = 3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    loss = log_loss(y_test, y_pred)
    print("log_loss: ", loss)

    f1 = f1_score(y_test, y_pred, average="micro")
    print("f1 score: ", f1)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: ", accuracy)
    return [f1, accuracy]

def k_clus(x_train, x_test, y_train, y_test):
    n = 100
    clu_model = KMeans(n_clusters=n)
    x_train_clu_index = clu_model.fit_predict(x_train)
    print("indices", x_train_clu_index)
    x_train_clusters = [[] for x in range(n)]
    for i in range(x_train.shape[0]):
        x_train_clusters[x_train_clu_index[i]].append(x_train.iloc[i])
    # print(x_train_cluster)
    x_test_clu_index = clu_model.predict(x_test)
    x_test_clusters = [[] for x in range(n)]
    for i in range(x_train.shape[0]):
        x_test_clusters[x_test_clu_index[i]].append(x_test.iloc[i])

def mlp(x_train, x_test, y_train, y_test=None):
    layers = (1280, 1024, 768, 576, 432, 207)
    model = MLPClassifier(hidden_layer_sizes=layers, random_state=1, max_iter=75, verbose=True).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # y_pred_prob = model.predict_proba(x_test)
    # loss = log_loss(y_test, y_pred_prob)
    # print("log_loss: ", loss)
    #
    # f1 = f1_score(y_test, y_pred, average="micro")
    # print("f1 score: ", f1)
    #
    # accuracy = accuracy_score(y_test, y_pred)
    # print("accuracy: ", accuracy)

    return y_pred


def main():
    x_train, y_train_scored, y_train_nonscored, x_test = load_files()
    y_sample = pd.read_csv("sample_submission.csv")
    print(y_sample.shape)

    x_train['cp_type'] = (x_train['cp_type'] == 'trt_cp').values.astype(int)
    x_train['cp_dose'] = (x_train['cp_dose'] == 'D1').values.astype(int)
    x_train = x_train.drop("sig_id", 1)
    y_train_scored = y_train_scored.drop("sig_id", 1)
    y_train_nonscored = y_train_nonscored.drop("sig_id", 1)
    x_test = x_test.drop("sig_id", 1)
    x_test['cp_type'] = (x_test['cp_type'] == 'trt_cp').values.astype(int)
    x_test['cp_dose'] = (x_test['cp_dose'] == 'D1').values.astype(int)
    # deal with data sparsity and imbalance

    # create linear model
    print("data obtain success")
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train_scored, random_state=0, test_size = 0.2)

    # y_train = y_train_scored
    print(x_train.shape)
    print(x_test.shape)
    print(y_train_nonscored.shape)
    pca = PCA(n_components=100)
    x_train = pca.fit_transform(x_train)
    print(x_train.shape)
    x_test = pca.transform(x_test)
    print(x_test.shape)
    print(x_train)
    # print(x_train.loc[0])


    # randomF(x_train, x_test, y_train, y_test)

    OneVsRest_SVC(x_train, x_test, y_train, y_test)
    # y_pred = mlp(x_train, x_test, y_train)
    # submission = pd.read_csv('sample_submission.csv')
    # headers = submission.columns.values
    # submission = submission.to_numpy()
    # for i in range(1, 207):
    #     submission[:, i] = y_pred[:, (i - 1)]
    # result = pd.DataFrame(data=submission, columns=headers)
    #
    # result.to_csv("pred_Submission.csv")

    # perform cross validation
    # scores = cross_validate(clf, x_train, y_train_scored, scoring = 'f1', cv = 5, return_estimator = True, average = "macro")
    # kf = KFold(n_splits=2)
    # kf.get_n_splits(x_train)
    avg_score = 0
    # for train_index, test_index in kf.split(x_train):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = x_train[train_index], x_train[test_index]
    #     y_train, y_test = y_train_scored[train_index], y_train_scored[test_index]



    # graph the model

    # predict y_train

if __name__ == '__main__':
    main()