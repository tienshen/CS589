import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, hamming_loss, log_loss, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def load_files():
    x_train = pd.read_csv("train_features.csv") #, nrows = 1500)
    y_train_scored = pd.read_csv("train_targets_scored.csv")#, nrows  = 1500)
    y_train_nonscored = pd.read_csv("train_targets_nonscored.csv")#, nrows  = 1500)
    x_test = pd.read_csv("test_features.csv")
    return x_train, y_train_scored, y_train_nonscored, x_test

def Log_Loss (model, x_test, y_test):
    probability = model.predict_proba(x_test)

    average_log_loss = 0
    for i in range(y_test.shape[1]):  # iterate through the columns
        average_log_loss += log_loss(y_test[:, i], probability[:, i], labels=[0, 1])

    average_log_loss /= y_test.shape[1]

    return average_log_loss

def main():

    x_train, y_train_scored, y_train_nonscored, x_test = load_files()
    x_train['cp_type'] = (x_train['cp_type'] == 'trt_cp').values.astype(int)
    x_train['cp_dose'] = (x_train['cp_dose'] == 'D1').values.astype(int)
    x_train = x_train.drop("sig_id", 1)
    x_test['cp_type'] = (x_test['cp_type'] == 'trt_cp').values.astype(int)
    x_test['cp_dose'] = (x_test['cp_dose'] == 'D1').values.astype(int)
    x_test = x_test.drop("sig_id", 1)
    y_train_scored = y_train_scored.drop("sig_id", 1)
    y_train_nonscored = y_train_nonscored.drop("sig_id", 1)
    y_train = y_train_scored
    print("data obtain success")
    pca = PCA(n_components=100)
    x_train = pca.fit_transform(x_train)

    x_test = pca.transform(x_test)

    #Logistic Regression
    submission = pd.read_csv('sample_submission.csv')
    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000)).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    headers = submission.columns.values
    submission = submission.to_numpy()
    for i in range(1, 207):
        submission[:, i] = y_pred[:, (i - 1)]
    result = pd.DataFrame(data=submission, columns=headers)

    result.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()