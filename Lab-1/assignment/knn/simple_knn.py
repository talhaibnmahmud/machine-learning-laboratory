""" A Simple KNN Implementation"""
from collections import Counter

import numpy as np
import numpy.typing as npt


def euclidean_distance(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> np.float64:
    """Calculate the Euclidean distance between two vectors.
    """
    return np.sqrt(np.sum((x - y) ** 2), dtype=np.float64)  # type: ignore


class SimpleKNN:
    """Simple KNN Class"""

    def __init__(self, n_neighbors: int):
        self.k = n_neighbors

    def fit(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.int8]) -> None:
        """Train Model"""
        self.x_train = x
        self.y_train = y

    def predict(self, test: npt.NDArray[np.float64]) -> npt.NDArray[np.int8]:
        """ Predict Result"""
        predicted_labels = [self._calculate(x) for x in test]
        return np.array(predicted_labels)   # type: ignore

    def _calculate(self, x: npt.NDArray[np.float64]) -> np.int8:
        # calculate the distance between x and each point in x_train
        distances = [euclidean_distance(x, sample) for sample in self.x_train]

        # find the k nearest points
        k_nearest_idx: npt.NDArray[np.float64] = np.argsort(distances)[:self.k] # type: ignore
        k_nearest_labels = [self.y_train[i] for i in k_nearest_idx]

        # return the most common class label
        return Counter(k_nearest_labels).most_common(1)[0][0]
