from .haarFeatures import HaarFeature
from .utils import WAE
import numpy as np


class WeakClassifier:

    def __init__(self, feature: HaarFeature, threshold: float, parity: int):
        self.feature: HaarFeature = feature
        self.threshold: float = threshold
        self.parity: int = parity

    def classify(self, ii, scale: float = 1.0):
        feature_value: float = self.feature.value(ii, scale)

        a: float = self.parity * feature_value
        b: float = self.parity * self.threshold
        return int(a < b)

    def error(self, X, y: np.ndarray[int], weights: np.ndarray[float]):
        preds: np.ndarray[int] = np.array(list(map(self.classify, X)))
        return np.sum([WAE(w, y, preds) for w in weights])


class VJClassifier:
    """Rapid Face Detection classifier based on the model
    described in the paper published by Paul Viola and 
    Michael Jones https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf.
    """

    def __init__(self, num_features: int, betas: np.ndarray[float]) -> None:
        # self.alphas: np.ndarray = np.array(np.log(1 / betas))
        self.alphas: np.ndarray = ...
        self.classifiers: np.ndarray[WeakClassifier] = ...
        self.weights: np.ndarray[float] = ...

    def classify(self, x):
        """Final Strong Classification"""
        sum_half_alphas: float = 0.5 * np.sum(self.alphas)
        preds: np.ndarray[float] = np.array(
            [h.classify(x) for h in self.classifiers])
        sum_weighted_classification: float = np.sum(self.alphas * preds)

        return int(sum_weighted_classification >= sum_half_alphas)

    def __update_weights(self,
                         errors: np.ndarray[float],
                         classifications: np.ndarray[int]) -> None:
        betas: np.ndarray[float] = errors / (1 - errors)
        self.weights[1:] = self.weights[:-1] * \
            np.power(betas, 1 - classifications)

    def train(self, X, y):
        
        # Get the errors for each weak classifier
        errors: np.ndarray[float] = np.array([
            h.error(X, y, w)
            for h, w in zip(self.classifiers, self.weights)
        ])
        
        # Select the classifier which produces the minimum error
        best_classifier = self.classifiers[np.argmin(errors)]

        # Obtain the predictions from the classifier and classify
        # them as correct 1 or incorrect 0
        predictions: np.ndarray[int] = [
            int(best_classifier.classify(xi) == yi)
            for xi, yi in zip(X, y)
        ]

        # Updated the weights according to the AdaBoost algorithm
        self.__update_weights(errors, predictions)
