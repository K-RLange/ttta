import itertools
import os
import pickle
import warnings
from multiprocessing.pool import ThreadPool as Pool
from math import floor
import faulthandler
import matplotlib
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from wordcloud import WordCloud
from scipy.sparse import csr_matrix
from .LDA.lda_gibbs import vanilla_gibbs_func, load_wk_mat_func, final_assignment_func, load_dk_mat_func
from typing import Union, List, Tuple, Callable, Set
from collections import Counter
from tqdm import tqdm
import scipy
import time
from itertools import chain
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from .topic_matching import TopicClusters
from ..preprocessing.preprocess import create_dtm, get_word_and_doc_vector
faulthandler.enable()


class LDAPrototype:
    def __init__(self, K: int, alpha: float = None, gamma: float = None, prototype: int = 10, topic_threshold: List[Union[int, float]] = None,
                 prototype_measure: Union[str, Callable] = "jaccard", min_count: int = 2, max_assign: bool = False, verbose: int = 1):
        """
        Implements the LDAPrototype model, which trains multiple LDA models and selects the best one based on similarity measures.
        The LDAs are trained in C.
        Args:
            K: Number of topics
            alpha: alpha parameter for the LDA
            gamma: gamma parameter for the LDA
            prototype: Number of prototypes to train. If 1, the model is equivalent to a normal LDA.
            topic_threshold: Threshold for the minimum number of occurrences in a topic for a word to be relevant for the prototype selection
            prototype_measure: Measure to use for prototype selection
            min_count: Minimum number of occurrences for a word to be included in the vocabulary
            max_assign: Should the final assignment of the trained LDAs be chosen by maximum assignment from all previous training iterations
                        or should it be chosen randomly based on the word-topic probabilities (default)?
            verbose: Verbosity level. 0 does not print anything, 1 prints runtime-relevant information, 2 and higher
                     shows debugging information.
        """
        if not isinstance(K, int):
            try:
                K = int(K)
            except ValueError:
                raise TypeError("K must be an integer!")
        if K < 2:
            raise ValueError("K must be a natural number greater than 1!")
        if not isinstance(alpha, float) and not isinstance(alpha, np.ndarray) and alpha is not None:
            try:
                alpha = float(alpha)
            except ValueError:
                try:
                    alpha = np.array(alpha)
                except ValueError:
                    raise TypeError("alpha must be a float or an array of length K!")
        if isinstance(alpha, np.ndarray):
            if alpha.shape[0] != K:
                raise ValueError("alpha must be a float or an array of length K!")
            if np.any(alpha < 0):
                raise ValueError("alpha must only contain floats greater than 0!")
        if isinstance(alpha, float) and alpha < 0:
            raise ValueError("alpha must be a float greater than 0!")
        if not isinstance(gamma, float) and not isinstance(gamma, np.ndarray) and gamma is not None:
            try:
                gamma = float(gamma)
            except ValueError:
                try:
                    gamma = np.array(gamma)
                except ValueError:
                    raise TypeError("gamma must be a float or an array of length K!")
        if isinstance(gamma, np.ndarray):
            if gamma.shape[0] != K:
                raise ValueError("gamma must be a float or an array of length K!")
            if np.any(gamma < 0):
                raise ValueError("gamma must only contain floats greater than 0!")
        if isinstance(gamma, float) and gamma < 0:
            raise ValueError("gamma must be a float greater than 0!")
        if not isinstance(prototype, int):
            try:
                prototype = int(prototype)
            except ValueError:
                raise TypeError("prototype must be an integer!")
        if prototype < 1:
            raise ValueError("prototype must be a natural number greater than 0")
        if topic_threshold is None:
            topic_threshold = [5, 0.002]
        elif not isinstance(topic_threshold, list):
            raise TypeError("topic_threshold must be a list of an integer and a float!")
        elif topic_threshold[0] < 0 or topic_threshold[1] < 0:
            raise ValueError("topic_threshold must be a natural number greater than or equal to than 0")
        if prototype_measure not in ["jaccard", "cosine"] and not callable(prototype_measure):
            raise ValueError("prototype_measure must be either 'jaccard', 'cosine' or a callable!")
        if not isinstance(min_count, int):
            try:
                min_count = int(min_count)
            except ValueError:
                raise TypeError("min_count must be an integer!")
        if min_count < 1:
            raise ValueError("min_count must be a natural number greater than 0")
        if not isinstance(max_assign, bool):
            try:
                max_assign = bool(max_assign)
            except ValueError:
                raise TypeError("max_assign must be a boolean!")
        if not isinstance(verbose, float):
            try:
                verbose = float(verbose)
            except ValueError:
                raise TypeError("verbose must be a float!")
        self._K = K
        self._alpha = self._create_lda_parameters(alpha)
        self._gamma = self._create_lda_parameters(gamma)
        self._prototype = prototype
        self._max_assign = max_assign
        self._verbose = verbose
        self._assignments = None
        self._vocab = []
        self._dtm = None
        self._word_vec = None
        self._doc_vec = None
        self._document_topic_matrix = []
        self._threshold = topic_threshold
        self._min_count = min_count
        self._measure = prototype_measure
        self._is_trained = False
        self._trained_words = [0]
        self._len_of_docs = []
        self._deleted_indices = []
        self._last_end = [0]

    def _create_lda_parameters(self, param: Union[np.ndarray, float, None]) -> np.ndarray:
        """
        Creates a parameter List for LDA from a float, a numpy array or automatically from K
        Args:
            param: parameter - either alpha or gamma
        Returns:
            param: parameter list
        """
        if not isinstance(param, float) and not isinstance(param, np.ndarray) and param is not None:
            try:
                param = float(param)
            except ValueError:
                try:
                    param = np.array(param)
                except ValueError:
                    raise TypeError("param must be a float or an array of length K!")
        if isinstance(param, float) and param <= 0:
            raise ValueError("param must be a positive float!")
        if isinstance(param, np.ndarray):
            if param.shape[0] != self._K:
                raise ValueError("param must be a float or an array of length K!")
            if np.any(param <= 0):
                raise ValueError("param must only contain floats greater than 0!")
        if param is None:
            param = np.repeat(1 / self._K, self._K)
        elif isinstance(param, float):
            param = np.repeat(param, self._K)
        else:
            param = param.astype(float)
        return param

    def _create_dtm(self, texts: List[List[str]]) -> None:
        """
        Creates a document-term matrix from a list of texts and updates the existing dtm if there is one.
        Stores both the dtm and the vocabulary in the class.
        Args:
            texts: list of texts
        Returns:
            None
        """
        if self._verbose > 0:
            print("Creating document-term matrix...")
        self._dtm, self._vocab, self._deleted_indices = create_dtm(texts, self._vocab, self._min_count, self._deleted_indices, self._dtm)
        if self._verbose > 1:
            print(f"Created document-term matrix with shape {self._dtm.shape}")

    @staticmethod
    def _get_word_and_doc_vector(dtm: Union[csr_matrix, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Turns a document-term matrix into index vectors. The word vector contains the vocabulary index for each word
        occurrence including multiple occurrences in one text. The document vector contains the document index for each
        word occurrence including multiple occurrences in one document.
        Args:
            dtm: document-term matrix
        Returns:
            word_vec: word vector
            doc_vec: document vector
        """
        return get_word_and_doc_vector(dtm)

    def fit(self, texts: List[List[str]], epochs: int = 200, first_chunk: bool = True,
            chunk_end: int = None, memory_start: int = 0, workers=1) -> None:
        """
        LDA Gibbs Sampler that uses a C-implementation under the hood
        Args:
            texts: list of tokenized texts
            epochs: number of epochs to train the models for
            first_chunk: when using a RollingLDA, is this the first time chunk? Can be ignored, if no RollingLDA is used.
            chunk_end: end index of the period to train the model for when using a RollingLDA. Can be ignored, if no RollingLDA is used.
            memory_start: index to start loading the assignment vectors for a RollingLDA from. Can be ignored, if no RollingLDA is used.
            workers: number of workers to use for parallelization
        Returns:
            None
        """
        if not isinstance(texts, list):
            try:
                texts = list(texts)
            except ValueError:
                raise TypeError("texts must be a list!")
        if not isinstance(texts[0], list):
            try:
                texts = [list(x) for x in texts]
            except ValueError:
                raise TypeError("texts must be a list of lists!")
        if not isinstance(texts[0][0], str):
            try:
                texts = [[str(x) for x in text] for text in texts]
            except ValueError:
                raise TypeError("texts must be a list of lists of strings!")
        if not isinstance(epochs, int):
            try:
                epochs = int(epochs)
            except ValueError:
                raise TypeError("epochs must be an integer!")
        if epochs < 1:
            raise ValueError("epochs must be a natural number greater than 0")
        if not isinstance(first_chunk, bool):
            try:
                first_chunk = bool(first_chunk)
            except ValueError:
                raise TypeError("first_chunk must be a boolean!")
        if chunk_end is not None and not isinstance(chunk_end, int):
            try:
                chunk_end = int(chunk_end)
            except ValueError:
                raise TypeError("chunk_end must be an integer!")
        if chunk_end is not None and chunk_end < 0:
            raise ValueError("chunk_end must be a natural number greater than 0")
        if not isinstance(memory_start, int):
            try:
                memory_start = int(memory_start)
            except ValueError:
                raise TypeError("memory_start must be an integer!")
        if memory_start < 0:
            raise ValueError("memory_start must be a natural number greater than 0")
        if not isinstance(workers, int):
            try:
                workers = int(workers)
            except ValueError:
                raise TypeError("workers must be an integer!")
        if chunk_end is None:
            chunk_end = len(texts)
        texts = texts[self._last_end[-1]:chunk_end]
        old_dtm_size = self._dtm.shape[0] if self._dtm is not None else 0
        self._create_dtm(texts)

        word_vec, doc_vec = self._get_word_and_doc_vector(self._dtm[old_dtm_size:chunk_end+old_dtm_size, :])
        if self._verbose > 1:
            print("Word_vec length: ", len(word_vec))
        corpus_size = self._dtm.shape[0] - old_dtm_size
        vocab_size = len(self._vocab)

        if memory_start > 0:
            memory_start = sum([memory_start > x for x in self._deleted_indices])
            memory_start = self._len_of_docs[:memory_start].sum()
        if first_chunk:
            memory_word_topic_matrix = np.zeros((vocab_size, self._K), dtype=np.uint64)
        else:
            memory_word_vec = self._word_vec[memory_start:]
            memory_assignments = self._assignments[memory_start:]
            memory_word_topic_matrix = self.get_word_topic_matrix(memory_word_vec, memory_assignments)
            if self._verbose > 1:
                print(f"Loaded a memory of {len(memory_assignments)} assignments")

        document_topic_matrix = np.zeros((corpus_size, self._K), dtype=np.uint64)
        v_sum = np.sum(memory_word_topic_matrix, axis=0)

        self._calculate_prototype(word_vec, doc_vec, memory_word_topic_matrix,
                                  document_topic_matrix, v_sum, workers, epochs, old_dtm_size)
        self._is_trained = True
        self._doc_vec = np.concatenate((self._doc_vec, doc_vec), axis=0) if self._doc_vec is not None else doc_vec
        doc_id, tmp_counts = np.unique(self._doc_vec, return_counts=True)
        self._len_of_docs = np.zeros(np.array(self._doc_vec.max() + 1, dtype=np.uint64), dtype=np.uint32)
        self._len_of_docs[doc_id] = tmp_counts
        self._last_end.append(self._dtm.shape[0])

    def _calculate_prototype(self, word_vec: np.ndarray, doc_vec: np.ndarray, word_topic_matrix: np.ndarray,
                             document_topic_matrix: np.ndarray, v_sum: np.ndarray, workers: int, epochs: int = 200,
                             period_start: int = 0):
        """
        Calculates the LDAPrototype for the texts given in fit().
        Args:
            word_vec: vector of word ids in the vocabulary for each word
            doc_vec: vector of document ids for each word
            word_topic_matrix: word-topic matrix
            document_topic_matrix: document-topic matrix
            v_sum: sum of word-topic matrix
            workers: number of workers to use for parallelization
            epochs: number of epochs to train the models for
            period_start: index to start loading the assignment vectors for a RollingLDA from. Can be ignored, if no RollingLDA is used.
        """
        if not isinstance(word_vec, np.ndarray):
            raise TypeError("word_vec must be a numpy array!")
        if not isinstance(doc_vec, np.ndarray):
            raise TypeError("doc_vec must be a numpy array!")
        if not isinstance(word_topic_matrix, np.ndarray):
            raise TypeError("word_topic_matrix must be a numpy array!")
        if not isinstance(document_topic_matrix, np.ndarray):
            raise TypeError("document_topic_matrix must be a numpy array!")
        if not isinstance(v_sum, np.ndarray):
            raise TypeError("v_sum must be a numpy array!")
        if not isinstance(workers, int):
            raise TypeError("workers must be an integer!")
        if not isinstance(epochs, int):
            raise TypeError("epochs must be an integer!")
        if not isinstance(period_start, int):
            raise TypeError("period_start must be an integer!")
        if period_start < 0:
            raise ValueError("period_start must be a natural number greater than 0")

        all_results = []
        if workers > 1:
            assignments = np.random.randint(0, self._K, len(word_vec), dtype=np.uint32)
            with Pool(workers) as pool:
                all_results = pool.starmap(self._sample_gibbs, [(word_vec.copy(), assignments.copy(), doc_vec.copy(),
                                                                 word_topic_matrix.copy(), document_topic_matrix.copy(),
                                                                 v_sum.copy(), epochs) for _ in range(self._prototype)])
            print("results are here!")
            all_results = list(all_results)
        else:
            iterator = range(self._prototype)
            if self._verbose > 0:
                iterator = tqdm(iterator)
            for sample in iterator:
                if self._verbose > 0:
                    iterator.set_description(f"Prototype {sample + 1}/{self._prototype}")
                assignments = np.random.randint(0, self._K, len(word_vec), dtype=np.uint32)
                res = self._sample_gibbs(word_vec.copy(), assignments.copy(), doc_vec.copy(), word_topic_matrix.copy(), document_topic_matrix.copy(),
                                         v_sum.copy(), epochs)
                all_results.append(res)

        if self._prototype > 1:
            current_word_topic_matrices = [self.get_word_topic_matrix(word_vec, x[0]) for x in all_results]
            sclop = TopicClusters(current_word_topic_matrices, self._measure, self._threshold, self._K).select_prototype()
            if sclop is None:
                new_assignments = np.random.randint(0, self._K, len(word_vec), dtype=np.uint32)
                prototype = [new_assignments.copy(), self.get_word_topic_matrix(word_vec, new_assignments), self.get_document_topic_matrix(doc_vec, new_assignments)]
            else:
                prototype = all_results[np.argmax(sclop)]
        else:
            prototype = all_results[0]

        doc_vec += period_start
        self._assignments = np.concatenate((self._assignments, prototype[0]), axis=0) if self._assignments is not None else prototype[0]
        self._word_vec = np.concatenate((self._word_vec, word_vec), axis=0) if self._word_vec is not None else word_vec
        self._document_topic_matrix.append(prototype[2])

    def _sample_gibbs(self, word_vec: np.ndarray, assignment_vec: np.ndarray, doc_vec: np.ndarray, word_topic_matrix: np.ndarray,
                      document_topic_matrix: np.ndarray, v_sum: np.ndarray, epochs: int = 200) -> (
            Tuple)[np.ndarray, np.ndarray, np.ndarray]:
        """
        LDA Gibbs Sampler that uses a C-implementation under the hood and edits the inputs in place
        Args:
            word_vec: word vector
            assignment_vec: assignment vector
            doc_vec: document vector
            word_topic_matrix: word-topic matrix
            document_topic_matrix: document-topic matrix
            v_sum: sum of word-topic matrix
            epochs: number of epochs to train the models for
        Returns:
            assignment_vec: assignment vector
            word_topic_matrix: word-topic matrix
            document_topic_matrix: document-topic matrix
        """
        if not isinstance(word_vec, np.ndarray):
            raise TypeError("word_vec must be a numpy array!")
        if not isinstance(doc_vec, np.ndarray):
            raise TypeError("doc_vec must be a numpy array!")
        if not isinstance(word_topic_matrix, np.ndarray):
            raise TypeError("word_topic_matrix must be a numpy array!")
        if not isinstance(document_topic_matrix, np.ndarray):
            raise TypeError("document_topic_matrix must be a numpy array!")
        if not isinstance(v_sum, np.ndarray):
            raise TypeError("v_sum must be a numpy array!")
        if not isinstance(epochs, int):
            raise TypeError("epochs must be an integer!")
        vanilla_gibbs_func(word_vec, assignment_vec, doc_vec, word_topic_matrix, document_topic_matrix, v_sum,
                           np.array(self._alpha.copy()), np.array(self._gamma.copy()), self._K, epochs, 0)

        if self._max_assign:
            final_assignment_func(word_vec, assignment_vec, doc_vec, word_topic_matrix, document_topic_matrix, v_sum, self._alpha.copy(), self._gamma.copy(), self._K, epochs, 0)
        return assignment_vec, word_topic_matrix, document_topic_matrix

    def get_word_topic_matrix(self, word_vec: np.ndarray, assignment_vec: np.ndarray) -> np.ndarray:
        """
        Create a word-topic matrix from a word vector and an assignment vector
        Args:
            word_vec: word vector
            assignment_vec: assignment vector
        Returns:
            word_topic_matrix: word-topic matrix
        """
        if not isinstance(word_vec, np.ndarray):
            try:
                word_vec = np.array(word_vec, dtype=np.uint64)
            except ValueError:
                raise TypeError("word_vec must be a numpy array!")
        if not isinstance(assignment_vec, np.ndarray):
            try:
                assignment_vec = np.array(assignment_vec, dtype=np.uint32)
            except ValueError:
                raise TypeError("assignment_vec must be a numpy array!")
        word_topic_matrix = np.zeros((len(self._vocab), self._K), dtype=np.uint64)
        load_wk_mat_func(word_vec, assignment_vec, word_topic_matrix, self._K)
        return word_topic_matrix.copy()

    def get_document_topic_matrix(self, doc_vec: np.ndarray, assignment_vec: np.ndarray) -> np.ndarray:
        """
        Create a word-topic matrix from a word vector and an assignment vector
        Args:
            doc_vec: document vector
            assignment_vec: assignment vector
        Returns:
            document_topic_matrix: document-topic matrix
        """
        if not isinstance(doc_vec, np.ndarray):
            try:
                doc_vec = np.array(doc_vec, dtype=np.uint64)
            except ValueError:
                raise TypeError("doc_vec must be a numpy array!")
        if not isinstance(assignment_vec, np.ndarray):
            try:
                assignment_vec = np.array(assignment_vec, dtype=np.uint32)
            except ValueError:
                raise TypeError("assignment_vec must be a numpy array!")
        document_topic_matrix = np.zeros((len(set(doc_vec)), self._K), dtype=np.uint64)
        load_dk_mat_func(doc_vec, assignment_vec, document_topic_matrix, self._K)
        return document_topic_matrix.copy()

    def top_words(self, number: int = 5, topic: int = None, importance: Union[bool, np.ndarray] = True, word_topic_matrix: np.ndarray = None) -> Union[List[str], List[List[str]]]:
        """
        Get the top words for a topic
        Args:
            number: number of top words to return
            topic: topic to return top words for or None for all topics
            importance: bool: should the importance or absolute frequency be used to determine the top words
                        np.array: importance of words for each topic
            word_topic_matrix: word-topic matrix
        Returns:
            top_words: top words
        """
        if not isinstance(number, int):
            try:
                number = int(number)
            except ValueError:
                raise TypeError("number must be an integer!")
        if number < 1:
            raise ValueError("number must be a natural number greater than 0")
        if topic is not None and not isinstance(topic, int):
            try:
                topic = int(topic)
            except ValueError:
                raise TypeError("topic must be an integer!")
        if topic is not None and (topic < 1 or topic > self._K):
            raise ValueError("topic must be a natural number between 1 and K")
        if not isinstance(importance, bool) and not isinstance(importance, np.ndarray):
            try:
                importance = bool(importance)
            except ValueError:
                raise TypeError("importance must be a boolean!")
        if word_topic_matrix is not None and not isinstance(word_topic_matrix, np.ndarray):
            raise TypeError("word_topic_matrix must be a numpy array!")
        if word_topic_matrix is None:
            word_topic_matrix = self.get_word_topic_matrix(self._word_vec, self._assignments)
        if isinstance(importance, bool) and importance:
            importance = self._calculate_importance(word_topic_matrix) if importance else None
        if topic is None:
            return [self.top_words(number, k, importance, word_topic_matrix) for k in range(1, self._K + 1)]
        else:
            if isinstance(importance, np.ndarray):
                return [self._vocab[index] for index in np.argsort(-importance[topic - 1, :])[:number]]
            return [self._vocab[index] for index in np.argsort(-word_topic_matrix[topic - 1, :])[:number]]

    @staticmethod
    def _calculate_importance(word_topic_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate the importance of words for each topic, weighting down words frequently used in multiple topics
        Args:
            word_topic_matrix: word-topic matrix
        Returns:
            importance: importance of words for each topic
        """
        if not isinstance(word_topic_matrix, np.ndarray):
            raise TypeError("word_topic_matrix must be a numpy array!")
        word_topic_matrix = word_topic_matrix.transpose()
        importance = word_topic_matrix / np.sum(word_topic_matrix, axis=1)[:, np.newaxis]
        log_importance = np.log(importance + 1e-5)
        importance = importance * (log_importance - np.mean(log_importance, axis=0)[np.newaxis, :])
        return importance

    def wordclouds(self, topic: int = None, number: int = 50, path: str = "wordclouds.pdf", height: int = 600, width: int = 700,
                   show: bool = True, word_topic_matrix: np.ndarray = None) -> None:
        """
        Create a word cloud from a topic
        Args:
            topic: The topic to create a word cloud for or None for all topics
            number: number of top words to include in the word cloud
            path: path to a file to store the wordcloud files in
            width: width of the image
            height: height of the image
            show: should the image be shown
            word_topic_matrix: Optional. If you want the wordclouds to be based on a specific word-topic matrix, you can pass it here.
        Returns:
            None
        """
        if not isinstance(topic, int) and topic is not None:
            try:
                topic = int(topic)
            except ValueError:
                raise TypeError("topic must be an integer!")
        if topic is not None and (topic < 1 or topic > self._K):
            raise ValueError("topic must be a natural number between 1 and K")
        if not isinstance(number, int):
            try:
                number = int(number)
            except ValueError:
                raise TypeError("number must be an integer!")
        if number < 1:
            raise ValueError("number must be a natural number greater than 0")
        if path is not None and not isinstance(path, str):
            raise TypeError("path must be a string or None!")
        if not isinstance(height, int):
            try:
                height = int(height)
            except ValueError:
                raise TypeError("height must be an integer!")
        if height < 1:
            raise ValueError("height must be a natural number greater than 0")
        if not isinstance(width, int):
            try:
                width = int(width)
            except ValueError:
                raise TypeError("width must be an integer!")
        if width < 1:
            raise ValueError("width must be a natural number greater than 0")
        if not isinstance(show, bool):
            try:
                show = bool(show)
            except ValueError:
                raise TypeError("show must be a boolean!")
        if path is not None:
            matplotlib.use('Agg')
        if path is None and show is None:
            raise ValueError("'path' and 'show' cannot both be None")
        if word_topic_matrix is None:
            word_topic_matrix = self.get_word_topic_matrix(self._word_vec, self._assignments)
        importance = self._calculate_importance(word_topic_matrix)
        if topic is None:
            word_weights = [{self._vocab[index]: importance[k, index] for index in np.argsort(-importance[k, :])[:number]} for k in range(self._K)]
        else:
            word_weights = [{self._vocab[index]: importance[topic, index] for index in np.argsort(-importance[topic, :])[:number]}]
        figures = []
        for i, cloud in enumerate(word_weights):
            wordcloud = WordCloud(width=width, height=height, background_color='white').generate_from_frequencies(cloud)
            fig = plt.figure(figsize=(15, 15 * height / width), dpi=150)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            if topic is None:
                plt.title(f'Word Cloud {i + 1}')
            else:
                plt.title(f'Word Cloud {topic + 1}')
            figures.append(fig)
            if show:
                fig.show()
        if path is not None:
            with PdfPages(path) as pdf:
                for i, fig in enumerate(figures):
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

    def get_assignment_vec(self) -> np.ndarray:
        """
        Returns:
            assignment vector
        """
        return self._assignments.copy()

    def get_word_vec(self) -> np.ndarray:
        """
        Returns:
            word vector
        """
        return self._word_vec.copy()

    def get_doc_vec(self) -> np.ndarray:
        """
        Returns:
            document vector
        """
        return self._doc_vec.copy()

    def get_vocab(self) -> List[str]:
        """
        Returns:
            vocabulary
        """
        return self._vocab.copy()

    def get_params(self) -> dict:
        """
        Returns:
            all parameters as a dictionary
        """
        return self.__dict__.copy()

    def set_params(self, params: dict) -> None:
        """
        Sets all parameters given py a dictionary to the parameters of the same name in the class
        Returns:
            None
        """
        if not isinstance(params, dict):
            raise TypeError("parameters must be a dictionary!")
        self.__dict__.update(params)

    def save(self, path: str) -> None:
        """
        Saves the model to a pickle file
        Returns:
            None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        pickle.dump(self, open(path, "wb"))

    def load(self, path: str) -> None:
        """
        Loads a model from a pickle file
        Returns:
            None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        loaded = pickle.load(open(path, "rb"))
        self.__dict__ = loaded.__dict__

    def shrink_model(self, new_start_index: int) -> None:
        """
        Shrinks the model to a new start index to save memory
        """
        # todo implement
        raise NotImplementedError

    def is_trained(self) -> bool:
        """
        Returns:
            Boolean if the model has been trained before
        """
        return self._is_trained


