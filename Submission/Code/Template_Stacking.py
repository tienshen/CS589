# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import metrics
from sklearn import svm
# feel free to import any sklearn model here


def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 419
    # of testing samples: 150
    ------
    """
    df = pd.read_csv("../../Data/breast_cancer_data/data.csv")

    cols = df.columns
    X = df[cols[2:-1]].to_numpy()
    y = df[cols[1]].to_numpy()
    y = (y=='M').astype(np.int) * 2 - 1

    train_X = X[:-150]
    train_y = y[:-150]

    test_X = X[-150:]
    test_y = y[-150:]

    return train_X, train_y, test_X, test_y

def main():
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data()
    
    # Stacking models:
    # Create your stacked model using StackingClassifier
    estimators = [("ada", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), learning_rate=0.1, n_estimators=20)), ("rf", RandomForestClassifier(max_features="sqrt", n_estimators=20))]
    stc = StackingClassifier(estimators=estimators)
    # fit the model on the training data
    stc.fit(train_X, train_y)
    # Get and print f1-score on test data
    print(test_X)
    print(test_y)
    pred_y = stc.predict(test_X)
    f1 = metrics.f1_score(test_y, pred_y, average='weighted')
    print(f1)
if __name__ == '__main__':
    main()
