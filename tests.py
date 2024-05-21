import numpy as np

from tree_code import find_best_split, DecisionTree

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

    feature_types = ["real", "real", "real", "real", "real"]
    X = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9, 0.9, 0.9],
        ]
    )
    y = np.array([0, 1, 0, 1])

    tree = DecisionTree(feature_types=feature_types)
    tree.fit(X, y)

    predictions = tree.predict(X)
    print("Predictions:", predictions)
