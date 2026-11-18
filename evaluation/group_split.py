
import numpy as np


def group_cross_validation(X, y, groups, n_splits=5):
    """
    Group-aware cross-validation.

    Ensures that observations belonging to the
    same trip never appear in both train and validation.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y and groups must have equal lengths.")

    unique_groups = np.unique(groups)

    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError("Invalid number of splits.")

    # Divide unique trips into folds
    group_folds = np.array_split(unique_groups, n_splits)

    splits = []

    for validation_groups in group_folds:

        val_mask = np.isin(groups, validation_groups)
        train_mask = ~val_mask

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        splits.append((train_idx, val_idx))

    return splits
