"""Equal-weight composites of the top-k univariate features and drop-one-group ridge ablation."""

import numpy as np

from .common import daily_ic

TOP_K = (3, 5, 8)


def composite_predictions(X_train, y_train, dates_train, X_test, k_values=TOP_K):
    """Per-fold equal-weight composites; features are signed and picked by their training-fold mean IC.

    Returns {k: prediction} and the (signs, order) of features used.
    """
    ics = np.array([daily_ic(X_train[:, j], y_train, dates_train).mean() for j in range(X_train.shape[1])])
    order = np.argsort(-np.abs(ics))
    signs = np.sign(ics)
    preds = {k: (X_test[:, order[:k]] * signs[order[:k]]).mean(axis=1) for k in k_values}
    return preds, ics
