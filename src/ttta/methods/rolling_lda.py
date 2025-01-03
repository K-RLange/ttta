"""This module implements a rolling LDA model for diachronic topic modeling."""
import os
import pickle
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from typing import Union, List, Tuple, Callable
from tqdm import tqdm
from .lda_prototype import LDAPrototype
from datetime import datetime
import warnings
import seaborn as sns
from operator import itemgetter
import math
import webbrowser
from ..preprocessing import chunk_creation
from pyLDAvis import prepare, save_html, display, show


class RollingLDA:
    """Implements a rolling LDA model for diachronic topic modeling."""

    def __init__(self, K: int, how: Union[str, List[datetime]] = "ME", warmup: int = 48, memory: int = 3, alpha: float = None, gamma: float = None,
                 initial_epochs: int = 100, subsequent_epochs: int = 50, min_count: int = 2, max_assign=False, prototype: int = 10,
                 topic_threshold: List[Union[int, float]] = None, prototype_measure: Union[str, Callable] = "jaccard", lda: LDAPrototype = None,
                 min_docs_per_chunk: int = None, verbose: int = 1) -> None:
        """Initialize a RollingLDA model.

        Args:
            K: The number of topics.
            how: List of datetime dates indicating the end of time chunks or a string indicating the frequency of the time chunks as in pandas.resample().
            warmup: The number of chunks to use for the initial fitting period of RollinglDA.
            memory: The number of chunks to look back for when calculating the topic proportions for the RollingLDA.
            alpha: The alpha parameter of the LDA model.
            gamma: The gamma parameter of the LDA model.
            initial_epochs: The number of epochs to train the LDA model in the initial fit.
            subsequent_epochs: The number of epochs to train the LDA model in the subsequent fits.
            min_count: The minimum number of times a word must occur in the corpus to be included in the vocabulary.
            max_assign: Whether the maximum number of topic assignments per document should be used as the final topic assignment.
            prototype: Number of LDAs to fit and to choose a Prototype from.
            topic_threshold: The minimum occurrences within a topic to be relevant for the prototype similarity calculation.
            prototype_measure: The measure to use for calculating the distance between the LDAs. Can be "jaccard", "cosine" or a custom function.
            lda: An LDAPrototype model to use instead of fitting a new one.
            min_docs_per_chunk: The minimum number of documents a chunk must contain to be used for the LDA training.
            verbose: The verbosity of the output. 0 does not show any output, 1 shows a progress bar, 2 also shows information relevant for debugging.
        """
        if not isinstance(how, str) and not isinstance(how, list):
            raise TypeError("how must be a string or a list of datetime dates!")
        if not isinstance(warmup, int):
            try:
                if warmup == int(warmup):
                    warmup = int(warmup)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("warmup must be an integer!")
        if warmup < 0:
            raise ValueError("warmup must be a natural number")
        if not isinstance(memory, int):
            try:
                if memory == int(memory):
                    memory = int(memory)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("memory must be an integer!")
        if not isinstance(K, int):
            try:
                if K == int(K):
                    K = int(K)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("K must be an integer!")
        if memory < 1:
            raise ValueError("memory must be a natural number greater than 0")
        if not isinstance(initial_epochs, int):
            try:
                if initial_epochs == int(initial_epochs):
                    initial_epochs = int(initial_epochs)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("initial_epochs must be an integer!")
        if not isinstance(subsequent_epochs, int):
            try:
                if subsequent_epochs == int(subsequent_epochs):
                    subsequent_epochs = int(subsequent_epochs)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("subsequent_epochs must be an integer!")
        if initial_epochs < 1 or subsequent_epochs < 1:
            raise ValueError("initial_epochs and subsequent_epochs must be natural numbers greater than 0")
        if min_docs_per_chunk is None:
            min_docs_per_chunk = K * 10
        if not isinstance(min_docs_per_chunk, int):
            try:
                if min_docs_per_chunk == int(min_docs_per_chunk):
                    min_docs_per_chunk = int(min_docs_per_chunk)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("min_docs_per_chunk must be an integer or None!")
        if min_docs_per_chunk < 1:
            raise ValueError("min_docs_per_chunk must be a natural number greater than 0")
        self._min_docs_per_chunk = min_docs_per_chunk
        if not topic_threshold:
            topic_threshold = [5, 0.002]
        self._K = K
        self._how = how
        self._warmup = warmup
        self._memory = memory
        if alpha is None:
            self._alpha = 1 / self._K
        else:
            self._alpha = alpha
        if gamma is None:
            self._gamma = 1 / self._K
        else:
            self._gamma = gamma
        self._initial_epochs = initial_epochs
        self._subsequent_epochs = subsequent_epochs
        self._prototype = prototype
        self._threshold = topic_threshold
        self._measure = prototype_measure
        self._min_count = min_count
        self._max_assign = max_assign
        self.lda = lda if lda is not None else LDAPrototype(K=K, alpha=alpha, gamma=gamma, prototype=prototype, topic_threshold=topic_threshold,
                                                            prototype_measure=prototype_measure, min_count=min_count, max_assign=max_assign,
                                                            verbose=verbose-1)
        self._updated_how = []
        self._verbose = verbose
        self._distances_simulated = None
        self._distances_observed = None
        self.chunk_indices = None
        self._date_column = "date"
        self._text_column = "text"
        self._last_text = None

    def fit(self, texts: pd.DataFrame, workers: int = 1, text_column: str = "text", date_column: str = "date") -> None:
        """Fits a RollingLDA model in place to the given texts from scratch.

        When updating an existing model, use fit_update() instead.
        Args:
            texts: A pandas DataFrame containing the columns text_column and date_column containing the documents and their respective dates.
                   The dates must be in a format interpretable by pandas.to_datetime(). Each element of the text_column column must be a list of strings.
            workers: The number of workers to use for parallelization.
            text_column: The name of the column in texts containing the documents.
            date_column: The name of the column in texts containing the dates.
        Returns:
            None
        """
        if not isinstance(texts, pd.DataFrame):
            raise TypeError("texts must be a pandas DataFrame!")
        if not isinstance(text_column, str):
            raise TypeError("text_column must be a string!")
        if text_column not in texts.columns:
            raise ValueError("texts must contain the column specified in text_column!")
        if not isinstance(date_column, str):
            raise TypeError("date_column must be a string!")
        if date_column not in texts.columns:
            raise ValueError("texts must contain the column specified in date_column!")
        self._text_column = text_column
        self._date_column = date_column
        if not isinstance(texts[self._text_column].iloc[0], list):
            raise TypeError("The elements of the 'texts' column of texts must each contain a tokenized document as a list of strings!")
        if not isinstance(workers, int):
            try:
                if workers == int(workers):
                    workers = int(workers)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("workers must be an integer!")
        if self.lda.is_trained():
            raise ValueError("The LDA has already been trained. If you want to update the training process, use fit_update() instead!")

        texts[date_column] = pd.to_datetime(texts[date_column])
        texts.sort_values(by=date_column, inplace=True)
        self.chunk_indices = self._get_time_indices(texts)
        if self.chunk_indices.shape[0] < self._warmup + 1:
            raise ValueError(f"The number of chunks in the data must be larger than warmup!")
        iterator = self.chunk_indices.iloc[self._warmup:].iterrows()
        if self._verbose > 0:
            iterator = tqdm(iterator, unit="chunk")
        for i, row in iterator:
            if self._verbose and i < len(self.chunk_indices) - 1:
                iterator.set_description(f"Processing {int(self.chunk_indices.iloc[i + 1]['chunk_start'] - 1 - self.chunk_indices.iloc[i]['chunk_start'])} documents in "
                                         f"chunk {self.chunk_indices.iloc[i][date_column].strftime('%Y-%m-%d')}")
            if i == self._warmup:  # fit warmup chunks
                self.lda.fit(texts[text_column], epochs=self._initial_epochs,
                             chunk_end=self.chunk_indices.iloc[self._warmup + 1]["chunk_start"], memory_start=0, workers=workers)
                continue
            end = len(texts) if i + 1 >= len(self.chunk_indices) else self.chunk_indices.iloc[i + 1]["chunk_start"] - 1
            self.lda.fit(texts[text_column], epochs=self._subsequent_epochs, first_chunk=False, chunk_end=end, memory_start=row["memory_start"], workers=workers)
        if self._last_text is None:
            self._last_text = {date_column: None, "index": 0}
        self._last_text[date_column] = texts[date_column].iloc[-1]
        self._last_text["index"] += len(texts) - 1

    def fit_update(self, texts: pd.DataFrame, how: Union[str, List[datetime]] = None, workers: int = 1, text_column: str = "text",
                   date_column: str = "date") -> None:
        """Update the fit of a RollingLDA model in place to the given texts from scratch.
        When training a model from scratch, fit() is called instead.

        Args:
            texts: A pandas DataFrame containing the columns text_column and date_column containing the documents and their respective dates.
                   The dates must be in a format interpretable by pandas.to_datetime(). Each element of the text_column column must be a list of strings.
            how: List of datetime dates indicating the end of time chunks or a string indicating the frequency of the time chunks as in pandas.resample().
            workers: The number of workers to use for parallelization.
            text_column: The name of the column in texts containing the documents.
            date_column: The name of the column in texts containing the dates.
        Returns:
            None
        """
        if not isinstance(texts, pd.DataFrame):
            raise TypeError("texts must be a pandas DataFrame!")
        if not isinstance(text_column, str):
            raise TypeError("text_column must be a string!")
        if text_column not in texts.columns:
            raise ValueError("texts must contain the column specified in text_column!")
        if text_column != self._text_column:
            texts[text_column] = texts[self._text_column]
        if not isinstance(date_column, str):
            raise TypeError("date_column must be a string!")
        if date_column not in texts.columns:
            raise ValueError("texts must contain the column specified in date_column!")
        if date_column != self._date_column:
            texts[date_column] = texts[self._date_column]
        if not isinstance(texts[self._text_column].iloc[0], list):
            raise TypeError("The elements of the 'texts' column of texts must each contain a tokenized document as a list of strings!")
        if not isinstance(how, str) and not isinstance(how, list) and how is not None:
            raise TypeError("how must be a string or a list of datetime dates!")
        if not isinstance(workers, int):
            try:
                if workers == int(workers):
                    workers = int(workers)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("workers must be an integer!")
        if not self.lda.is_trained():
            warnings.warn("The LDA has not been trained yet. Will use fit() instead.")
            self.fit(texts, workers)
            return
        last_trained = len(self.chunk_indices)
        texts[self._date_column] = pd.to_datetime(texts[self._date_column])
        texts.sort_values(by=self._date_column, inplace=True)
        self.chunk_indices = self._get_time_indices(texts, update=True, how=how)
        iterator = self.chunk_indices.iloc[last_trained:].iterrows()
        if self._verbose > 0:
            iterator = tqdm(iterator, unit="chunk")
        for i, row in iterator:
            if self._verbose and i < len(self.chunk_indices) - 1:
                iterator.set_description(f"Processing {self.chunk_indices.iloc[i + 1]['chunk_start'] - 1 - self.chunk_indices.iloc[i]['chunk_start']} documents in "
                                         f"chunk {self.chunk_indices.iloc[i]['date'].strftime('%Y-%m-%d')}")
            end = self.chunk_indices.iloc[i + 1]["chunk_start"] if i + 1 < len(self.chunk_indices) else len(texts) + self._last_text["index"] + 1   # todo hier ueberpruefen, ob ohne self.text richtige indices berechnet werden
            self.lda._last_end[-1] = self.chunk_indices.iloc[i]["chunk_start"] - self.chunk_indices.iloc[last_trained]["chunk_start"]
            self.lda.fit(texts[self._text_column], epochs=self._subsequent_epochs, first_chunk=False, chunk_end=end - self._last_text["index"] - 1, memory_start=row["memory_start"], workers=workers)
        self._last_text[self._date_column] = texts[self._date_column].iloc[-1]
        self._last_text["index"] += len(texts)

    def top_words(self, chunk: Union[int, str] = None, topic: int = None, number: int = 5, importance: bool = True,
                  return_as_data_frame: bool = True) -> Union[List[str], List[List[str]]]:
        """Return the top words for the given chunk and topic.

        Args:
            chunk: The chunk for which the top words should be returned. If None, the top words over the entire time frame are returned.
                   If "all", the top words for each individual chunk are returned.
            topic: The topic for which the top words should be returned. If None, the top words for all topics are returned.
            number: The number of top words to return.
            importance: Whether the words should be weighted based on their importance to a topic or their absolute frequency.
            return_as_data_frame: Whether the top words should be returned as a pandas DataFrame.
        """
        if not isinstance(chunk, int) and chunk is not None and chunk != "all":
            try:
                if chunk == int(chunk):
                    chunk = int(chunk)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("chunk must be an integer, 'all' or None!")
        if not isinstance(topic, int) and topic is not None:
            try:
                if topic == int(topic):
                    topic = int(topic)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("topic must be an integer or None!")
        if not isinstance(number, int):
            try:
                if number == int(number):
                    number = int(number)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("number must be an integer!")
        if not isinstance(importance, bool):
            raise TypeError("importance must be a boolean!")
        if chunk is None:
            word_topic_matrix = self.get_word_topic_matrix()
            return self.lda.top_words(number=number, topic=topic, importance=importance, word_topic_matrix=word_topic_matrix,
                                      return_as_data_frame=return_as_data_frame)
        elif chunk == "all":
            top_words = [self.top_words(chunk=i, number=number, importance=importance, return_as_data_frame=return_as_data_frame) for i in range(len(self.chunk_indices))]
            if return_as_data_frame:
                top_words = pd.concat(top_words)
                top_words.index = [f"Chunk {i+1}, word {x+1}" for i in range(len(self.chunk_indices)) for x in range(number)]
            return top_words
        else:
            word_topic_matrix = self.get_word_topic_matrix(chunk)
            return self.lda.top_words(number=number, topic=topic, importance=importance, word_topic_matrix=word_topic_matrix,
                                      return_as_data_frame=return_as_data_frame)

    def get_word_topic_matrix(self, chunk: int = None) -> np.ndarray:
        """Return the topic assignments for the given chunk.

        If no chunk is given, the topic assignments for all chunks are returned.
        Args:
            chunk: The chunk for which the topic assignments should be returned. If None, the topic assignments for all chunks are returned.
        Returns:
            The topic assignments for the given chunk or all chunks.
        """
        if not isinstance(chunk, int) and chunk is not None:
            try:
                if chunk == int(chunk):
                    chunk = int(chunk)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("chunk must be an integer or None!")
        assignments = self.lda.get_assignment_vec()
        words = self.lda.get_word_vec()
        len_of_docs = self.lda._len_of_docs
        if chunk is None:
            return self.lda.get_word_topic_matrix(words, assignments)
        else:
            start_index = len_of_docs[:self.chunk_indices.iloc[chunk]["chunk_start"]].sum()
            if chunk == len(self.chunk_indices) - 1:
                end_index = len(assignments)
            else:
                end_index = len_of_docs[:self.chunk_indices.iloc[chunk + 1]["chunk_start"]].sum()
            return self.lda.get_word_topic_matrix(words[start_index:end_index], assignments[start_index:end_index])

    def get_parameters(self) -> dict:
        """Return the parameters of the RollingLDA model.

        Returns:
            A dictionary containing the parameters of the RollingLDA model.
        """
        return self.__dict__.copy()

    def set_parameters(self, parameters: dict) -> None:
        """Set the parameters of the RollingLDA model.

        Args:
            parameters: A dictionary containing the parameters of the RollingLDA model.
        Returns:
            None
        """
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary!")
        self.__dict__.update(parameters)

    def save(self, path: str) -> None:
        """Save the RollingLDA model to the given path as a .pickle-file.

        Args:
            path: The path to which the RollingLDA model should be saved.
        Returns:
            None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        pickle.dump(self, open(path, "wb"))

    def load(self, path: str) -> None:
        """Load a pickled RollingLDA model from the given path.

        Args:
            path: The path from which the RollingLDA model should be loaded.
        Returns:
            None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        loaded = pickle.load(open(path, "rb"))
        self.__dict__ = loaded.__dict__

    def get_document_topic_matrix(self, chunk: int = None) -> np.ndarray:
        """Return the document-topic matrix for the given chunk.

        If no chunk is given, the document-topic matrix for all chunks is returned.
        Args:
            chunk: The chunk for which the document-topic matrix should be returned. If None, the document-topic matrix for all chunks is returned.
        Returns:
            The document-topic matrix for the given chunk or all chunks.
        """
        if not isinstance(chunk, int) and chunk is not None:
            try:
                if chunk == int(chunk):
                    chunk = int(chunk)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("chunk must be an integer or None!")
        if chunk is None:
            all_matrices = []
            for i in range(len(self.chunk_indices)):
                all_matrices.append(self.get_document_topic_matrix(i))
            return np.concatenate(all_matrices, axis=0)
        else:
            assignments = self.lda.get_assignment_vec()
            docs = self.lda.get_doc_vec()
            len_of_docs = self.lda._len_of_docs
            start_index = len_of_docs[:self.chunk_indices.iloc[chunk]["chunk_start"]].sum()
            if chunk == len(self.chunk_indices) - 1:
                end_index = len(assignments)
            else:
                end_index = len_of_docs[:self.chunk_indices.iloc[chunk + 1]["chunk_start"]].sum()
            return self.lda.get_document_topic_matrix(docs[start_index:end_index],
                                                      assignments[start_index:end_index])


    def _get_time_indices(self, texts: pd.DataFrame, update: bool = False, how: Union[str, List[datetime]] = None) -> pd.DataFrame:
        """Create the time indices for the given texts.

        If update is True, the time indices are appended to the existing ones.
        Args:
            texts: A pandas DataFrame containing the documents and their respective dates.
                   The dates must be in a format interpretable by pandas.to_datetime(). The texts must be a list of strings.
            update: Whether the time indices should be appended to the existing ones.
            how: List of datetime dates indicating the end of time chunks or a string indicating the frequency of the time chunks. Used to create time
                 chunks when fixed dates were used for the initial fit. If None, the same time chunk rule as in the initial fit is used.
        """
        if not isinstance(texts, pd.DataFrame):
            raise TypeError("texts must be a pandas DataFrame!")
        if not isinstance(update, bool):
            raise TypeError("update must be a boolean!")
        if how is None:
            how = self._how
        elif isinstance(how, str):
            if isinstance(self._how, list):
                warnings.warn(f"The time indices are created using periodic distances instead of fixed dates. This might create inconsistencies.")
            elif self._how != how:
                warnings.warn(f"The time indices are created using {how} instead of {self._how}. This might create inconsistencies.")
        else:
            if isinstance(self._how, str):
                warnings.warn(f"The time indices are created using fixed dates instead of periodic distances. This might create inconsistencies.")
        last_date = self._last_text[self._date_column] if self.lda.is_trained() else None
        period_start = chunk_creation._get_time_indices(texts, how, last_date=last_date, date_column=self._date_column,
                                                        min_docs_per_chunk=self._min_docs_per_chunk)
        if update:
            period_start["chunk_start"] += self._last_text["index"] + 1
            memory_start = [period_start["chunk_start"].iloc[i - self._memory] if i - self._memory >= 0 else self.chunk_indices["chunk_start"].iloc[i - self._memory] for i in range(len(period_start))]
        else:
            memory_start = [0 if i <= self._warmup else period_start["chunk_start"].iloc[max(i - self._memory, 0)] for i in range(len(period_start))]
        period_start["memory_start"] = memory_start
        period_start = self.chunk_indices._append(period_start) if update else period_start
        period_start = period_start.reset_index(drop=True)

        if isinstance(how, list) and isinstance(self._how, list):
            self._how.append(how)
        elif isinstance(how, str) and isinstance(self._how, str):
            if how != self._how:
                self._updated_how.append(how)
        else:
            self._updated_how.append(how)
        return period_start


    def wordclouds(self, chunks: List[int] = None, topic: int = None, number: int = 50, path: str = "wordclouds",
                   height: int = 500, width: int = 700, show: bool = True):
        """Plot the wordclouds for the given chunk.

        If chunk is None, the wordclouds are plotted for every time chunk.
        Args:
            chunks: The chunks for which the wordclouds should be plotted. If None, the wordclouds over the entire time frame are plotted.
            topic: The topic to create a word cloud for or None for all topics
            number: number of top words to include in the word cloud
            path: path to a directory to store the wordcloud files in
            width: width of the image
            height: height of the image
            show: should the image be shown
        Returns:
            None
        """
        if not isinstance(chunks, List) and chunks is not None:
            raise TypeError("chunks must be a list or None!")
        if chunks and not all([isinstance(x, int) for x in chunks]):
            raise TypeError("chunks must be a list of integers!")
        if chunks is None:
            chunks = range(len(self.chunk_indices))
        for chunk in chunks:
            word_topic_matrix = self.get_word_topic_matrix(chunk)
            if path is not None and not os.path.exists(path):
                os.makedirs(path)
                path += "/" if path[-1] != "/" else ""
            self.lda.wordclouds(topic=topic, number=number, path=os.path.join(path, f"chunk_{chunk}.pdf"), height=height, width=width,
                                show=show, word_topic_matrix=word_topic_matrix)

    def visualize(self, chunk: int = None, number: int = 30,
                  path: str = "ldaviz.html", open_browser: bool = True) \
            -> None:
        """
        Visualize the RollingLDA model using pyLDAvis.
        Args:
            chunk: The chunk for which the visualization should be created.
                   If None, the visualization is based on all chunks.
            number: The number of top words to include in the visualization.
            path: The path to save the visualization to.
            open_browser: Whether to open the visualization in the browser.

        Returns:
            None
        """
        if not isinstance(chunk, int) and chunk is not None and chunk != "all":
            try:
                if chunk == int(chunk):
                    chunk = int(chunk)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("chunk must be an integer, 'all' or None!")
        if not isinstance(number, int):
            try:
                if number == int(number):
                    number = int(number)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("number must be an integer!")
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        if not isinstance(open_browser, bool):
            raise TypeError("open_browser must be a boolean!")
        word_topic_matrix = self.get_word_topic_matrix(chunk)
        topic_term_dists = (word_topic_matrix /
                            word_topic_matrix.sum(axis=0, keepdims=True)).transpose()
        document_topic_matrix = self.get_document_topic_matrix(chunk)
        doc_topic_dists = (document_topic_matrix /
                            document_topic_matrix.sum(axis=1,
                                                      keepdims=True))
        term_frequency = word_topic_matrix.sum(axis=1)
        len_of_docs = self.lda._len_of_docs
        vocab = self.lda.get_vocab()
        if chunk is not None:
            if chunk == len(self.chunk_indices):
                len_of_docs = len_of_docs[self.chunk_indices["chunk_start"].iloc[chunk]:]
            else:
                len_of_docs = len_of_docs[self.chunk_indices["chunk_start"].iloc[chunk]:self.chunk_indices["chunk_start"].iloc[chunk+1]]
        ldaviz_data = prepare(topic_term_dists, doc_topic_dists, len_of_docs, vocab, term_frequency, number)
        save_html(ldaviz_data, path)
        if open_browser:
            webbrowser.open_new_tab(path)
