
import numpy as np

from evaluation.group_split import group_cross_validation


def test_no_group_leakage():

    # Simulated data from 10 different trips
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    # Each trip contributes 10 observations
    groups = np.repeat(np.arange(10), 10)

    splits = group_cross_validation(
        X, y, groups, n_splits=5
    )

    for train_idx, val_idx in splits:

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])

        # No trip should appear in both sets
        assert train_groups.isdisjoint(val_groups)

    print("Group leakage test passed!")


if __name__ == "__main__":
    test_no_group_leakage()
