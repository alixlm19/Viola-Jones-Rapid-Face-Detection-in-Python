import numpy as np


def initialize_weights(
    n: int,
    num_negative: int,
    num_positive: int) -> np.ndarray[float]:
    """Initialize the classifier's weights
    
    Parameters:
    -----------
    n: `int`
        number of elements in the training set 
    num_negative: `int`
        number of negative examples in the training set
    num_positive: `int`
        number of positive examples in the training set
    """

    w1: float = 1 / (2 * num_negative)
    w2: float = 1 / (2 * num_positive)

    weights = np.zeros((n, 2))
    weights[0] = w1, w2

    return weights

@np.vectorize
def WAE(
    weights: np.ndarray[float],
    true: np.ndarray[int],
    pred: np.ndarray[int]) -> float:
    """Returns the Weighted Absolute Error
    
    Parameters:
    -----------
    weights: `np.ndarray[float]`
        Weights assigned to each weak classifier
    true: `np.ndarray[int]`
        True classification values
    pred: `np.ndarray[int]`
        Predicted classification values
    """
    return np.sum(weights * np.abs(true - pred))