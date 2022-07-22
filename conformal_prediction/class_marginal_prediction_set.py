import numpy as np


def compute_quantiles(X, Y, alpha, tau):
    n = X.shape[0]
    sis_tau = None
    Q = np.sort(sis_tau)[np.floor(alpha * (n + 1))]  # vector of Q_y-s for each y value


def compute_prediction_set(x_test, Y, tau, alpha):
    Q = compute_quantile(X, Y, alpha, tau)
