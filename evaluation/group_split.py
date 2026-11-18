
from sklearn.model_selection import GroupKFold


def group_cross_validation(X, y, groups, n_splits=5):
    """
    Generate group-aware cross-validation splits.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target labels.
    groups : array-like
        Trip ID for each training sample.
    n_splits : int
        Number of folds.

    Returns
    -------
    splits : list
        List of training and validation indices.
    """

    splitter = GroupKFold(n_splits=n_splits)

    splits = []

    for train_idx, val_idx in splitter.split(X, y, groups):
        splits.append((train_idx, val_idx))

    return splits
