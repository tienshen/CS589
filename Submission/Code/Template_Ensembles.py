# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 63
    # of testing samples: 20
    ------
    """
    train_X = np.genfromtxt("../../Data/gene_data/gene_train_x.csv", delimiter= ",")
    train_y = np.genfromtxt("../../Data/gene_data/gene_train_y.csv", delimiter= ",")
    test_X = np.genfromtxt("../../Data/gene_data/gene_test_x.csv", delimiter= ",")
    test_y = np.genfromtxt("../../Data/gene_data/gene_test_y.csv", delimiter= ",")

    return train_X, train_y, test_X, test_y
def RFC_run(N_list, train_X, train_y, test_X, test_y):
    n_samples, n_features = train_X.shape
    rf_sqrt = []
    rf_n = []
    rf_3 = []
    for N in N_list:
        # Train RF with m = sqrt(n_features) recording the errors (errors will be of size 150)
        rf = RandomForestClassifier(max_features="sqrt", n_estimators=N)
        rf.fit(train_X, train_y)
        acc = 1 - rf.score(test_X, test_y)
        rf_sqrt.append(acc)
        # Train RF with m = n_features recording the errors (errors will be of size 150)
        rf = RandomForestClassifier(max_features=None, n_estimators=N)
        rf.fit(train_X, train_y)
        acc = 1 - rf.score(test_X, test_y)
        rf_n.append(acc)

        # Train RF with m = n_features/3 recording the errors (errors will be of size 150)
        rf = RandomForestClassifier(max_features=int(n_features/3), n_estimators=N)
        rf.fit(train_X, train_y)
        acc = 1 - rf.score(test_X, test_y)
        rf_3.append(acc)
    # plot the Random Forest results
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_title("Number of Estimators vs Error (RandomForest)", color='C0')

    ax.plot(N_list, rf_sqrt, label='m = sqrt(n_features)')
    ax.plot(N_list, rf_n, label='m = n_features')
    ax.plot(N_list, rf_3, label='m = n_features/3')
    plt.xlabel('number of estimators')
    plt.ylabel('error')
    ax.legend()
    plt.show()
    pass

def ABC_run(N_list, train_X, train_y, test_X, test_y):
    ada_d1 = []
    ada_d3 = []
    ada_d5 = []
    # Train AdaBoost with max_depth = 1 recording the errors (errors will be of size 150)
    # ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), learning_rate=0.1, n_estimators=150)
    # ada.fit(train_X, train_y)
    # acc = 1 - ada.score(test_X, test_y)
    # print(acc)

    for N in N_list:
        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), learning_rate=0.1, n_estimators=N)
        ada.fit(train_X, train_y)
        acc = 1 - ada.score(test_X, test_y)
        ada_d1.append(acc)
        # Train AdaBoost with max_depth = 3 recording the errors (errors will be of size 150)
        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), learning_rate=0.1, n_estimators=N)
        ada.fit(train_X, train_y)
        acc = 1 - ada.score(test_X, test_y)
        ada_d3.append(acc)
        # Train AdaBoost with max_depth = 5 recording the errors (errors will be of size 150)
        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), learning_rate=0.1, n_estimators=N)
        ada.fit(train_X, train_y)
        acc = 1 - ada.score(test_X, test_y)
        ada_d5.append(acc)
    # plot the adaboost results
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_title("Number of Estimators vs Error (Adaboost)", color='C0')

    ax.plot(N_list, ada_d1, label='max_depth = 1')
    ax.plot(N_list, ada_d3, label='max_depth = 3')
    ax.plot(N_list, ada_d5, label='max_depth = 5')
    plt.xlabel('number of estimators')
    plt.ylabel('error')
    ax.legend()
    plt.show()

def main():
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data()
    # N_list = 150 # Each part will be tried with 1 to 150 estimators
    N_list = [(i+1) for i in range(50)]

    RFC_run(N_list, train_X, train_y, test_X, test_y)
    ABC_run(N_list, train_X, train_y, test_X, test_y)



if __name__ == '__main__':
    main()
