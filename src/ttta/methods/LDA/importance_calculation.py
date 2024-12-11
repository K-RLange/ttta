import numpy as np
"""Implements the calculation of word importance for topics in LDA."""
def calculate_importance(word_topic_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the importance of words for each topic, weighting down words frequently used in multiple topics.

    Args:
        word_topic_matrix: word-topic matrix
    Returns:
        importance: importance of words for each topic
    """
    if not isinstance(word_topic_matrix, np.ndarray):
        raise TypeError("word_topic_matrix must be a numpy array!")
    word_topic_matrix = word_topic_matrix.transpose()
    denominator = np.sum(word_topic_matrix, axis=1)[:, np.newaxis]
    denominator[denominator == 0] = 1
    importance = word_topic_matrix / denominator
    log_importance = np.log(importance + 1e-5)
    importance = importance * (log_importance - np.mean(log_importance, axis=0)[np.newaxis, :])
    return importance
