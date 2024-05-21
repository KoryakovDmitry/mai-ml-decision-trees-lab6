import numpy as np

from tree_code import find_best_split

if __name__ == "__main__":
    feature_vector = np.array([2, 3, 10])
    target_vector = np.array([0, 1, 0])
    thresholds, ginis, threshold_best, gini_best = find_best_split(
        feature_vector, target_vector
    )
    print("Thresholds:", thresholds)
    print("Ginis:", ginis)
    print("Best Threshold:", threshold_best)
    print("Best Gini:", gini_best)
