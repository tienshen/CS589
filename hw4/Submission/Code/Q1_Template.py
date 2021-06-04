import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt

def load_data(dataset):
    """
    Load a pair of data X,y 

    Params
    ------
    dataset:    train/valid/test

    Return
    ------
    X:          shape (N, 240)
    y:          shape (N, 1)
    """
    X = pd.read_csv(f"../../Data/housing_data/{dataset}_x.csv", header=None).to_numpy()
    y = pd.read_csv(f"../../Data/housing_data/{dataset}_y.csv", header=None).to_numpy()

    return X,y

def score(model, X, y):
    """
    Score the model with X, y

    Params
    ------
    model:  the model to predict with
    X:      the data to score on
    y:      the true value y

    Return
    ------
    mae:    the mean absolute error
    """
    
    pass


def hyper_parameter_tuning(model_class, param_grid, train, valid):
    """
    Tune the hyper-parameter using training and validation data

    Params
    ------
    model_class:    the model class
    param_grid:     the hyper-parameter grid, dict
    train:          the training data (train_X, train_y)
    valid:          the validatation data (valid_X, valid_y)

    Return
    ------
    model:          model fit with best params
    best_param:     the best params
    """
    train_X, train_y = train
    valid_X, valid_y = valid

    # Set up the parameter grid
    param_grid = list(ParameterGrid(param_grid))
    print(param_grid)
    # train the model with each parameter setting in the grid
    mae_lst = []
    for i in param_grid:
        model_class.set_params(**i)
        model_class.fit(train_X, train_y)
        pred = model_class.predict(valid_X)
        mae_lst.append(mean_absolute_error(valid_y, pred))
        print(i, mean_absolute_error(valid_y, pred))
    print(mae_lst.index(min(mae_lst)))
    # choose the model with lowest MAE on validation set
    best_params = param_grid[mae_lst.index(min(mae_lst))]
    # then fit the model with the training and validation set (refit)
    train_X = np.concatenate([train[0], valid[0]], axis=0)
    train_y = np.concatenate([train[1], valid[1]], axis=0)
    model_class.set_params(**best_params)
    model_class.fit(train_X, train_y)
    # return the fitted model and the best parameter setting
    return [model_class, best_params]

def plot_mae_alpha(model_class, params, train, valid, test, title="Model"):
    """
    Plot the model MAE vs Alpha (regularization constant)

    Params
    ------
    model_class:    The model class to fit and plot
    params:         The best params found 
    train:          The training dataset
    valid:          The validation dataest
    test:           The testing dataset
    title:          The plot title

    Return
    ------
    None
    """
    train_X = np.concatenate([train[0], valid[0]], axis=0)
    train_y = np.concatenate([train[1], valid[1]], axis=0)

    # set up the list of alphas to train on
    alpha = [0.1, 10, 30, 50, 70, 90, 120, 150]
    # train the model with each alpha, log MAE
    mae_lst = []
    for i in alpha:
        model_class.set_params(alpha=i, max_iter=1000)
        model_class.fit(train_X,train_y)
        pred = model_class.predict(test[0])
        mae_lst.append(mean_absolute_error(test[1], pred))
    # mae_lst = np.log(mae_lst)
    # plot the MAE - Alpha
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_title("Alpha vs MAE ({})".format(title), color='C0')

    ax.plot(alpha, mae_lst)
    plt.xlabel('Alpha')
    plt.ylabel('log MAE')
    ax.legend()

def main():
    """
    Load in data
    """
    train = load_data('train')
    valid = load_data('valid')
    test = load_data('test')

    """
    Define the parameter grid each each classifier
    e.g. lasso_grid = dict(alpha=[0.1, 0.2, 0.4],
                           max_iter=[1000, 2000, 5000])
    """
    lasso_grid = dict(alpha=[0.1, 0.5, 1,  5, 10, 30, 40, 50],
                      max_iter=[1000, 2000, 5000])
    ridge_grid = dict(alpha=[0.1, 0.5, 1, 5, 10, 30, 40, 50],
                      max_iter=[1000, 2000, 5000])
    ols = LinearRegression().fit(train[0], train[1])
    ols_pred = ols.predict(valid[0])
    print(mean_absolute_error(valid[1], ols_pred))
    rid = Ridge(normalize=False)
    las = Lasso(normalize=False)
    # Tune the hyper-paramter by calling the hyper-parameter tuning function
    # e.g. lasso_model, lasso_param = hyper_parameter_tuning(Lasso, lasso_grid, train, valid)
    las_model, las_param = hyper_parameter_tuning(las, lasso_grid, train, valid)
    rid_model, rid_param = hyper_parameter_tuning(rid, ridge_grid, train, valid)
    las_pred = las_model.predict(test[0])
    rid_pred = rid_model.predict(test[0])
    print("best parameters for Lasso:", las_param, "and MAE:", mean_absolute_error(test[1], las_pred))
    print("best parameters for Ridge:", rid_param, "and MAE:", mean_absolute_error(test[1], rid_pred))

    # Plot the MAE - Alpha plot by calling the plot_mae_alpha function
    # e.g. plot_mae_alpha(Lasso, lasso_param, train, valid, test, "Lasso")
    # plot_mae_alpha(las, las_param, train, valid, test, "lasso")
    plot_mae_alpha(rid, rid_param, train, valid, test, "ridge")
    plt.show()

if __name__ == '__main__':
    main()
