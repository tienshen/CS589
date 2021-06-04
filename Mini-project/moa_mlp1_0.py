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
    x_train = pd.read_csv("../input/lish-moa/train_features.csv")#, nrows = 1500)
    y_train_scored = pd.read_csv("../input/lish-moa/train_targets_scored.csv")#, nrows  = 1500)
    y_train_nonscored = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")#, nrows  = 1500)
    x_test = pd.read_csv("../input/lish-moa/test_features.csv")
    return x_train, y_train_scored, y_train_nonscored, x_test



def mlp(x_train, x_test, y_train, y_test=None):
    layers = (1280, 1024, 768, 576, 432, 207)
    model = MLPClassifier(hidden_layer_sizes=layers, random_state=1, max_iter=50, verbose=True).fit(x_train, y_train)
    y_pred = model.predict(x_test)


    return y_pred


def main():
    x_train, y_train_scored, y_train_nonscored, x_test = load_files()

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
    y_train = y_train_scored

    pca = PCA(n_components=100)
    x_train = pca.fit_transform(x_train)

    x_test = pca.transform(x_test)
    submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
    y_pred = mlp(x_train, x_test, y_train)
    headers = submission.columns.values
    submission = submission.to_numpy()
    for i in range(1, 207):
        submission[:, i] = y_pred[:, (i - 1)]
    result = pd.DataFrame(data=submission, columns=headers)

    result.to_csv("submission.csv", index=False)

 
if __name__ == '__main__':
    main()