"""Implements a prototype selection for topic models based on topic clustering."""
import numpy as np
from typing import Union, List, Tuple, Set
from scipy.cluster.hierarchy import linkage
import warnings
from math import floor
import scipy
import pandas as pd


class TopicClusters:
    """Given a measure, cluster topics from several topic models and return a score or each individual model pair."""

    def __init__(self, word_topic_matrices: List[np.ndarray] = None, measure: Union[str, callable] = "jaccard",
                 topic_threshold: List[Union[int, float]] = None, K: int = 30) -> None:
        """This class is used to cluster topics from topic models based on a given measure.
        It returns a score for each individual model pair, which can be used to select a prototype from the list of models.
        Apart from that, it can also give a summary as to which models have which topics in common.

        Args:
            word_topic_matrices: list of word-topic matrices
            measure: measure to use for prototype selection
            threshold_dict: dictionary containing the indices of the words that occur more than the threshold in a topic
        """
        if not isinstance(word_topic_matrices, list):
            raise TypeError("word_topic_matrices must be a list!")
        if not isinstance(word_topic_matrices[0], np.ndarray):
            raise TypeError("word_topic_matrices must be a list of numpy arrays!")
        if len(word_topic_matrices) < 2:
            raise ValueError("word_topic_matrices must contain at least two word-topic matrices!")
        self._word_topic_matrices = word_topic_matrices
        if not isinstance(measure, str) and not callable(measure):
            raise TypeError("measure must be a string or a callable!")
        self._measure = measure
        if topic_threshold is None:
            topic_threshold = [5, 0.002]
        elif not isinstance(topic_threshold, list):
            raise TypeError("topic_threshold must be a list of an integer and a float!")
        elif topic_threshold[0] < 0 or topic_threshold[1] < 0 or topic_threshold[1] > 1:
            raise ValueError("topic_threshold must contain a positive integer and a float between 0 and 1!")
        self._threshold = topic_threshold
        if not isinstance(K, int):
            try:
                if K == int(K):
                    K = int(K)
                else:
                    raise ValueError
            except:
                raise TypeError("K must be an integer!")
        elif K < 2:
            raise ValueError("K must be a natural number greater than 1")
        self._K = K
        if word_topic_matrices[0].shape[1] != K:
            raise ValueError("The number of word-topic matrices must be a multiple of K!")
        self._number_of_models = len(word_topic_matrices)
        self._vocab_length = word_topic_matrices[0].shape[0]
        self.sclops = None
        self.matched_topics = None

    def select_prototype(self) -> Union[np.ndarray, None]:
        """Select the prototype from a list of word-topic matrices.

        Returns:
            index of the prototype in the list of word-topic matrices
        """
        if self.sclops is None:
            self._compare_topics(self._word_topic_matrices, self._measure)
        return self.sclops if self.sclops is None else np.argmax([np.mean(x) for x in self.sclops])

    def get_matched_topics(self) -> Union[pd.DataFrame, None]:
        """
        Return a dataframe containing all matched topics for a list of word-topic matrices.

        Returns:
            dataframe containing all matched topics for a list of word-topic matrices
        """
        if self.matched_topics is None:
            self._compare_topics(self._word_topic_matrices, self._measure)
        return self.matched_topics

    def _compare_topics(self, word_topic_matrices: List[np.ndarray], measure: Union[str, callable]) -> None:
        """Compares the topics of the list of word-topic matrices, given a measure.
        Can be used to select a prototype or to give a summary of common topics.

        Args:
            word_topic_matrices: list of word-topic matrices
            measure: measure to use for prototype selection
        Returns:
            index of the prototype in the list of word-topic matrices
        """
        word_topic_matrices = np.vstack([x.transpose() for x in word_topic_matrices])
        threshold_dict, broken_indices = self._check_topic_thresholds(word_topic_matrices) if measure == "jaccard" else (None, set())
        if len(broken_indices) == self._number_of_models:
            warnings.warn(f"All models are broken. This can happen when there are too few documents to "
                          f"properly train the LDA or the words occur not often enough to satisfy the topic_threshold. Please consider"
                          f"lowering the values in topic_threshold. The assignments will be randomly sampled.")
            return None
        elif len(broken_indices) > 0:
            warnings.warn(f"Encountered {len(broken_indices)} broken models. This can happen when there are too few "
                          f"documents to properly train the LDA or the 'topic_threshold'-parameter has been set too "
                          f"restrictively. Please consider lowering the values in topic_threshold. "
                          f"The broken models will be ignored for the prototype selection.")
        sclops = [[] for _ in range(self._number_of_models)]
        matched_topics = None
        for i in range(self._number_of_models - 1):
            for j in range(i + 1, self._number_of_models):
                if i in broken_indices or j in broken_indices:
                    sclops[i].append(-1)
                    sclops[j].append(-1)
                    continue
                temp_matched_topics, sclop = self._cluster_topics(i, j, measure, threshold_dict)
                if matched_topics is None:
                    matched_topics = temp_matched_topics
                else:
                    matched_topics = pd.concat([matched_topics, temp_matched_topics])
                sclops[i].append(sclop)
                sclops[j].append(sclop)
        self.sclops = sclops
        self.matched_topics = matched_topics

    def _check_topic_thresholds(self, word_topic_matrices: np.ndarray) -> Tuple[dict, Set[int]]:
        """Filter out words that occur less than the threshold in a topic for the prototype selection.

        Args:
            word_topic_matrices: list of word-topic matrices
        Returns:
            threshold_dict: dictionary containing the indices of the words that occur more than the threshold in a topic
            broken_models: Index of all broken models that are to be ignored, because they store all words into one topic
                           (happens rarely when there are too few documents).
        """
        row_sums = np.sum(word_topic_matrices, axis=1)
        row_sums = np.where(row_sums < 1, 1, row_sums)
        vk_thresholds = np.argwhere((word_topic_matrices >= self._threshold[0]) & (word_topic_matrices / row_sums[:, None] >= self._threshold[1]))
        threshold_dict = {x: list(set(vk_thresholds[np.argwhere(vk_thresholds[:, 0] == x).flatten(), 1])) for x in range(self._number_of_models * self._K)}
        zero_indices = np.array([key for key, value in threshold_dict.items() if len(value) == 0])
        broken_models = set([floor(index / self._K) for index in zero_indices])
        return threshold_dict, broken_models

    def _cluster_topics(self, i: int, j: int, measure: Union[str, callable], threshold_dict: dict) -> Tuple[pd.DataFrame, float]:
        """
        Cluster topics given their word-topic matrices and returns the portion of matched pairs stemming from different models.

        Args:
            i: index of the first topic
            j: index of the second topic
            measure: measure to use for prototype selection
            threshold_dict: dictionary containing the indices of the words that occur more than the threshold in a topic
        Returns:
            portion of matched pairs stemming from different models
        """
        if not isinstance(i, int):
            raise TypeError("i must be an integer!")
        if not isinstance(j, int):
            raise TypeError("j must be an integer!")
        if not isinstance(threshold_dict, dict):
            raise TypeError("threshold_dict must be a dictionary!")
        if not isinstance(measure, str) and not callable(measure):
            raise TypeError("measure must be a string or a callable!")
        temp_mat = np.zeros((2 * self._K, self._vocab_length), dtype=int)
        if measure == "jaccard":
            for row in range(self._K):
                temp_mat[row, threshold_dict[row + i * self._K]] = 1
                temp_mat[row + self._K, threshold_dict[row + j * self._K]] = 1

        temp_distances = scipy.spatial.distance.pdist(temp_mat, metric=measure)
        linkage_result = linkage(temp_distances, method='complete')
        linkage_result = pd.DataFrame(linkage_result[:, :2], columns=['topic_one', 'topic_two']).astype(int)
        linkage_result["model_one"] = i
        linkage_result["model_two"] = j
        linkage_result = linkage_result[linkage_result['topic_one'] < 2 * self._K]
        linkage_result = linkage_result[linkage_result['topic_two'] < 2 * self._K]
        matched_topics = linkage_result[[(linkage_result['topic_one'][i] < self._K <= linkage_result['topic_two'][i]) or
                                                       (linkage_result['topic_two'][i] < self._K <= linkage_result['topic_one'][i])
                                                       for i, _ in linkage_result.iterrows()]]
        number_of_matched_topics = len(matched_topics)
        return matched_topics, 1 - (self._K - number_of_matched_topics) / (2 * self._K)