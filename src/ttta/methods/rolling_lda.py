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
from ..preprocessing import chunk_creation

class RollingLDA:
    def __init__(self, K: int, how: Union[str, List[datetime]] = "M", warmup: int = 48, memory: int = 3, alpha: float = None, gamma: float = None,
                 initial_epochs: int = 100, subsequent_epochs: int = 50, min_count: int = 2, max_assign=False, prototype: int = 10,
                 topic_threshold: List[Union[int, float]] = None, prototype_measure: Union[str, Callable] = "jaccard", lda: LDAPrototype = None,
                 min_docs_per_chunk: int = None, verbose: int = 1) -> None:
        """
            Initializes a RollingLDA model.
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
                warmup = int(warmup)
            except ValueError:
                raise TypeError("warmup must be an integer!")
        if warmup < 1:
            raise ValueError("warmup must be a natural number greater than 0")
        if not isinstance(memory, int):
            try:
                memory = int(memory)
            except ValueError:
                raise TypeError("memory must be an integer!")
        if memory < 1:
            raise ValueError("memory must be a natural number greater than 0")
        if not isinstance(initial_epochs, int):
            try:
                initial_epochs = int(initial_epochs)
            except ValueError:
                raise TypeError("initial_epochs must be an integer!")
        if not isinstance(subsequent_epochs, int):
            try:
                subsequent_epochs = int(subsequent_epochs)
            except ValueError:
                raise TypeError("subsequent_epochs must be an integer!")
        if initial_epochs < 1 or subsequent_epochs < 1:
            raise ValueError("initial_epochs and subsequent_epochs must be natural numbers greater than 0")
        if not isinstance(min_docs_per_chunk, int) and min_docs_per_chunk is not None:
            raise TypeError("min_docs_per_chunk must be an integer or None!")
        if min_docs_per_chunk is None:
            self.min_docs_per_chunk = K * 10
        else:
            self.min_docs_per_chunk = min_docs_per_chunk
        if not isinstance(verbose, float):
            try:
                verbose = float(verbose)
            except ValueError:
                raise TypeError("verbose must be a float!")

        self._K = K
        self._how = how
        self._warmup = warmup
        self._memory = memory
        if alpha is None:
            self._alpha = 1 / self._K
        if gamma is None:
            self._gamma = 1 / self._K
        self._initial_epochs = initial_epochs
        self._subsequent_epochs = subsequent_epochs
        self._prototype = prototype
        self._topic_threshold = topic_threshold
        self._prototype_measure = prototype_measure
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
        self._last_text = {self._date_column: None, "index": 0}

    def fit(self, texts: pd.DataFrame, workers: int = 1, text_column: str = "text", date_column: str = "date") -> None:
        """
        Fits a RollingLDA model in place to the given texts from scratch. When updating an existing model, use fit_update() instead.
        Args:
            texts: A pandas DataFrame containing the columns text_column and date_column containing the documents and their respective dates.
                   The dates must be in a format interpretable by pandas.to_datetime(). Each element of the text_column column must be a list of strings.
            workers: The number of workers to use for parallelization.
            text_column: The name of the column in texts containing the documents.
            date_column: The name of the column in texts containing the dates.
        Returns:
            None
        """
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
        if not isinstance(texts, pd.DataFrame):
            raise TypeError("texts must be a pandas DataFrame!")
        if text_column not in texts.columns or date_column not in texts.columns:
            raise ValueError("texts must contain the columns 'texts' and 'date'!")
        if not isinstance(texts[self._text_column].iloc[0], list):
            raise TypeError("The elements of the 'texts' column of texts must each contain a tokenized document as a list of strings!")
        if not isinstance(workers, int):
            try:
                workers = int(workers)
            except ValueError:
                raise TypeError("workers must be an integer!")
        if self.lda.is_trained():
            raise ValueError("The LDA has already been trained. If you want to update the training process, use fit_update() instead!")

        texts[date_column] = pd.to_datetime(texts[date_column])
        texts.sort_values(by=date_column, inplace=True)
        self.chunk_indices = self._get_time_indices(texts)
        self.lda.fit(texts[text_column], epochs=self._initial_epochs,
                     chunk_end=self.chunk_indices.iloc[self._warmup + 1]["chunk_start"], memory_start=0, workers=workers)
        iterator = self.chunk_indices.iloc[self._warmup + 1:].iterrows()
        if self._verbose > 0:
            iterator = tqdm(iterator, unit="chunk")
        for i, row in iterator:
            if self._verbose and i < len(self.chunk_indices) - 1:
                iterator.set_description(f"Processing {self.chunk_indices.iloc[i + 1]['chunk_start'] - 1 - self.chunk_indices.iloc[i]['chunk_start']} documents in "
                                         f"chunk {self.chunk_indices.iloc[i]['date'].strftime('%Y-%m-%d')}")
            end = len(texts) if i + 1 >= len(self.chunk_indices) else self.chunk_indices.iloc[i + 1]["chunk_start"] - 1
            self.lda.fit(texts[text_column], epochs=self._subsequent_epochs, first_chunk=False, chunk_end=end, memory_start=row["memory_start"], workers=workers)
        self.chunk_indices["chunk_start_preprocessed"] = [sum([x < y for x in self.lda._deleted_indices]) for y in self.chunk_indices["chunk_start"]]
        self._last_text[date_column] = texts[date_column].iloc[-1]
        self._last_text["index"] += len(texts) - 1

    def fit_update(self, texts: pd.DataFrame, how: Union[str, List[datetime]] = None, workers: int = 1, text_column: str = "text",
                   date_column: str = "date") -> None:
        """
        Updates the fit of a RollingLDA model in place to the given texts from scratch. When training a model from scratch, fit() is called instead.
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
        if not isinstance(texts, pd.DataFrame):
            raise TypeError("texts must be a pandas DataFrame!")
        if self._text_column not in texts.columns or self._date_column not in texts.columns:
            raise ValueError("texts must contain the columns 'texts' and 'date'!")
        if not isinstance(texts[self._text_column].iloc[0], list):
            raise TypeError("The elements of the 'texts' column of texts must each contain a tokenized document as a list of strings!")
        if not isinstance(how, str) and not isinstance(how, list) and how is not None:
            raise TypeError("how must be a string or a list of datetime dates!")
        if not isinstance(workers, int):
            try:
                workers = int(workers)
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
            self.chunk_indices["chunk_start_preprocessed"] = [sum([x < y for x in self.lda._deleted_indices]) for y in self.chunk_indices["chunk_start"]]
            self.lda._last_end[-1] = self.chunk_indices.iloc[i]["chunk_start_preprocessed"] - self.chunk_indices.iloc[last_trained]["chunk_start_preprocessed"]
            self.lda.fit(texts[self._text_column], epochs=self._subsequent_epochs, first_chunk=False, chunk_end=end - self._last_text["index"] - 1, memory_start=row["memory_start"], workers=workers)
        self._last_text[self._date_column] = texts[self._date_column].iloc[-1]
        self._last_text["index"] += len(texts)
        self.chunk_indices["chunk_start_preprocessed"] = [sum([x < y for x in self.lda._deleted_indices]) for y in self.chunk_indices["chunk_start"]]

    def topical_changes(self, mixture: float = 0.9, reference_period: int = 4, quantile_threshold: float = 0.99, samples: int = 500, save_path: str = None, load_path: str = None,
                        plot: bool = True, word_impact: bool = True, word_impact_number: int = 5, date_format: str = "%Y-%m-%d", fast: bool = True, reset: bool = False,
                        **plot_args) -> Union[pd.DataFrame, None]:
        """
            Calculates topical change points for the given parameters.
            Args:
                mixture: The mixture parameter as an a-priori-estimator for expected change in word usage from one time-chunk to the next. Lower values
                         will detect more change points, but the changes will be smaller. Higher values will detect fewer change points, but the changes
                         will be more noticeable.
                reference_period: The maximum number of chunks to look back for when calculating the topic proportions in the simulation. This can make
                                  the simulation more robust to outliers and allows for incorporating seasonal effects, e.g. when including the last four
                                  chunks when making a quarterly analysis. When using a high value, the change detection will detect gradual changes, since
                                  it compares the new chunks to very old chunks and when using a low value, the change detection will detect sudden changes,
                                  since it compares the new chunks to very recent chunks.
                quantile_threshold: The bootstrap quantile to compare the observed chunk similarities to, to detect significant changes in word usage.
                samples: The number of samples to draw for the simulation.
                save_path: The path to a directory in which the simulated topic proportions should be saved. If None, the simulated topic proportions are not saved.
                load_path: The path to a directory from which the simulated topic proportions should be loaded. If None, the simulated topic proportions are not loaded.
                plot: Whether the distances should be plotted.
                word_impact: Whether the leave one out word impact should be calculated, allowing for a more efficient interpretation of all changes.
                word_impact_number: The number of words to return for the word impact.
                date_format: The date format to return alongside the changes given in the word impact.
                fast: Whether the word impact should be calculated fast. If True, words that do not occur in the compared time chunks are ignored during
                      the word impact calculation. This should not change the results, unless "word_impact_number" is set to a high value compared to the
                      vocabulary size.
                reset: Whether the changes should be recalculated.
                **plot_args: Additional arguments to be passed to the matplotlib.pyplot.subplot function if plot is True, e.g. nrows, ncols and figsize.
            Returns:
                If word_impact is True, a pandas DataFrame containing the word impact for each topic and change point. Else None.
        """
        if not isinstance(mixture, float):
            try:
                mixture = float(mixture)
            except ValueError:
                raise TypeError("mixture must be a float!")
        if mixture <= 0 or mixture >= 1:
            raise ValueError("mixture must be a float between 0 and 1!")
        if not isinstance(reference_period, int):
            try:
                reference_period = int(reference_period)
            except ValueError:
                raise TypeError("reference_period must be an integer!")
        if reference_period < 1:
            raise ValueError("reference_period must be a natural number greater than 0!")
        if not isinstance(quantile_threshold, float):
            try:
                quantile_threshold = float(quantile_threshold)
            except ValueError:
                raise TypeError("quantile_threshold must be a float!")
        if quantile_threshold <= 0 or quantile_threshold >= 1:
            raise ValueError("quantile_threshold must be a float between 0 and 1!")
        if not isinstance(samples, int):
            try:
                samples = int(samples)
            except ValueError:
                raise TypeError("samples must be an integer!")
        if samples < 1:
            raise ValueError("samples must be a natural number greater than 0!")
        if not isinstance(save_path, str) and save_path is not None or not isinstance(load_path, str) and load_path is not None:
            raise TypeError("save_path and load_path must be string or None!")
        if not isinstance(plot, bool):
            try:
                plot = bool(plot)
            except ValueError:
                raise TypeError("plot must be a boolean!")
        if not isinstance(word_impact, bool):
            try:
                word_impact = bool(word_impact)
            except ValueError:
                raise TypeError("word_impact must be a boolean!")
        if not isinstance(word_impact_number, int):
            try:
                word_impact_number = int(word_impact_number)
            except ValueError:
                raise TypeError("word_impact_number must be an integer!")
        if word_impact_number < 1:
            raise ValueError("word_impact_number must be a natural number greater than 0!")
        if not isinstance(date_format, str):
            raise TypeError("date_format must be a string!")
        if not isinstance(fast, bool):
            try:
                fast = bool(fast)
            except ValueError:
                raise TypeError("fast must be a boolean!")
        if not isinstance(reset, bool):
            try:
                reset = bool(reset)
            except ValueError:
                raise TypeError("reset must be a boolean!")

        if self._verbose:
            print("Simulating distances")
        if self._distances_simulated is None or self._distances_observed is None or reset:
            self._simulate_changes(mixture, reference_period, quantile_threshold, samples, save_path, load_path)
        if plot:
            self.plot_distances(**plot_args)
        if self._verbose:
            print("Calculating leave-one-out word impacts")
        if word_impact:
            return self._word_impact(number=word_impact_number, reference_period=reference_period, date_format=date_format, fast=fast)

    def top_words(self, chunk: Union[int, str] = None, topic: int = None, number: int = 5, importance: bool = True) -> Union[List[str], List[List[str]]]:
        """
            Returns the top words for the given chunk and topic.
            Args:
                chunk: The chunk for which the top words should be returned. If None, the top words over the entire time frame are returned.
                       If "all", the top words for each individual chunk are returned.
                topic: The topic for which the top words should be returned. If None, the top words for all topics are returned.
                number: The number of top words to return.
                importance: Whether the words should be weighted based on their importance to a topic or their absolute frequency.

        """
        if not isinstance(chunk, int) and chunk is not None and chunk != "all":
            try:
                chunk = int(chunk)
            except ValueError:
                raise TypeError("chunk must be an integer, 'all' or None!")
        if not isinstance(topic, int) and topic is not None:
            try:
                topic = int(topic)
            except ValueError:
                raise TypeError("topic must be an integer or None!")
        if not isinstance(number, int):
            try:
                number = int(number)
            except ValueError:
                raise TypeError("number must be an integer!")
        if not isinstance(importance, bool):
            try:
                importance = bool(importance)
            except ValueError:
                raise TypeError("importance must be a boolean!")

        if chunk is None:
            word_topic_matrix = self.get_word_topic_matrix()
            return self.lda.top_words(number=number, topic=topic, importance=importance, word_topic_matrix=word_topic_matrix)
        elif chunk == "all":
            return [self.top_words(chunk=i, number=number, importance=importance) for i in range(len(self.chunk_indices))]
        else:
            word_topic_matrix = self.get_word_topic_matrix(chunk)
            return self.lda.top_words(number=number, topic=topic, importance=importance, word_topic_matrix=word_topic_matrix)

    def get_word_topic_matrix(self, chunk: int = None) -> np.ndarray:
        """
            Returns the topic assignments for the given chunk. If no chunk is given, the topic assignments for all chunks are returned.
            Args:
                chunk: The chunk for which the topic assignments should be returned. If None, the topic assignments for all chunks are returned.
            Returns:
                The topic assignments for the given chunk or all chunks.
        """
        if not isinstance(chunk, int) and chunk is not None:
            try:
                chunk = int(chunk)
            except ValueError:
                raise TypeError("chunk must be an integer or None!")
        assignments = self.lda.get_assignment_vec()
        words = self.lda.get_word_vec()
        len_of_docs = self.lda._len_of_docs
        if chunk is None:
            return self.lda.get_word_topic_matrix(words, assignments)
        else:
            start_index = len_of_docs[:self.chunk_indices.iloc[chunk]["chunk_start_preprocessed"]].sum()
            if chunk == len(self.chunk_indices) - 1:
                end_index = len(assignments)
            else:
                end_index = len_of_docs[:self.chunk_indices.iloc[chunk + 1]["chunk_start_preprocessed"]].sum()
            return self.lda.get_word_topic_matrix(words[start_index:end_index], assignments[start_index:end_index])

    def get_parameters(self) -> dict:
        """
            Returns the parameters of the RollingLDA model.
            Returns:
                A dictionary containing the parameters of the RollingLDA model.
        """
        return self.__dict__.copy()

    def set_parameters(self, parameters: dict) -> None:
        """
            Sets the parameters of the RollingLDA model.
            Args:
                parameters: A dictionary containing the parameters of the RollingLDA model.
            Returns:
                None
        """
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary!")
        self.__dict__.update(parameters)

    def save(self, path: str) -> None:
        """
            Saves the RollingLDA model to the given path as a .pickle-file.
            Args:
                path: The path to which the RollingLDA model should be saved.
            Returns:
                None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        pickle.dump(self, open(path, "wb"))

    def load(self, path: str) -> None:
        """
            Loads a pickled RollingLDA model from the given path.
            Args:
                path: The path from which the RollingLDA model should be loaded.
            Returns:
                None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        loaded = pickle.load(open(path, "rb"))
        self.__dict__ = loaded.__dict__


    def _get_time_indices(self, texts: pd.DataFrame, update: bool = False, how: Union[str, List[datetime]] = None) -> pd.DataFrame:
        """
            Creates the time indices for the given texts. If update is True, the time indices are appended to the existing ones.
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
            try:
                update = bool(update)
            except ValueError:
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
                                                        min_docs_per_chunk=self.min_docs_per_chunk)
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

    def _word_impact(self, number: int = 5, reference_period: int = 4, date_format: str = "%Y-%m-%d", fast: bool = True) -> pd.DataFrame:
        """
            Calculates the leave one out word impact for each topic and change point.
            Args:
                number: The number of words to return for the word impact.
                reference_period: The maximum number of chunks to look back for when calculating the topic proportions in the simulation. This can make
                                  the simulation more robust to outliers and allows for incorporating seasonal effects, e.g. when including the last four
                                  chunks when making a quarterly analysis. When using a high value, the change detection will detect gradual changes, since
                                  it compares the new chunks to very old chunks and when using a low value, the change detection will detect sudden changes,
                                  since it compares the new chunks to very recent chunks.
                date_format: The date format to return alongside the changes.
                fast: Whether the word impact should be calculated fast. If True, words that do not occur in the compared time chunks are ignored during
                      the word impact calculation. This should not change the results, unless "number" is set to a high value compared to the
                      vocabulary size.
            Returns:
                A pandas DataFrame containing the date and word impact for each topic and change point.
        """
        if not isinstance(number, int):
            try:
                number = int(number)
            except ValueError:
                raise TypeError("number must be an integer!")
        if number < 1:
            raise ValueError("number must be a natural number greater than 0")
        if not isinstance(reference_period, int):
            try:
                reference_period = int(reference_period)
            except ValueError:
                raise TypeError("reference_period must be an integer!")
        if reference_period < 1:
            raise ValueError("reference_period must be a natural number greater than 0")
        if not isinstance(date_format, str):
            raise TypeError("date_format must be a string!")
        if not isinstance(fast, bool):
            try:
                fast = bool(fast)
            except ValueError:
                raise TypeError("fast must be a boolean!")

        assignments = np.array([self.get_word_topic_matrix(chunk=chunk).transpose() for chunk, row in self.chunk_indices.iterrows()])
        topics = assignments.transpose((1, 2, 0))
        change_indices = [np.argwhere(self._distances_simulated.transpose()[x] < self._distances_observed.transpose()[x]).transpose(1, 0)[0].tolist() for x in
                          range(len(self._distances_simulated.transpose()))]
        events_end_date = [[self.chunk_indices[self._date_column].iloc[y] for y in x] for x in change_indices]
        leave_one_out_word_impact = {"Topic": [], "Date": [], "Significant Words": []}
        for k, changes in enumerate(change_indices):
            if len(changes) == 0:
                continue
            for i, change in enumerate(changes):
                temp = topics[k, :, max(1, change - self.runs[change][k], change - reference_period):change].sum(axis=1)
                loo_temp = np.array(
                    [0 if fast and topics[k, word, change] == 0 and temp[word] == 0 else cosine(np.delete(topics[k, :, change], word), np.delete(temp, word)) for word in range(topics.shape[1])])
                leave_one_out_substracted = loo_temp - cosine(topics[k, :, change], temp)
                significant_words = itemgetter(*leave_one_out_substracted.argsort()[-number:].astype(np.uint64).tolist())(self.lda.get_vocab())
                leave_one_out_word_impact["Topic"].append(k)
                leave_one_out_word_impact["Date"].append(events_end_date[k][i].strftime(date_format))
                leave_one_out_word_impact["Significant Words"].append(significant_words)
        leave_one_out_word_impact = pd.DataFrame(leave_one_out_word_impact)
        # todo loo richtig?
        return leave_one_out_word_impact

    def plot_distances(self, **plot_args) -> None:
        """
            Plots the distances between the observed and simulated topic proportions.
            Args:
                **plot_args: Additional arguments to be passed to the matplotlib.pyplot.subplot function, e.g. nrows, ncols and figsize.
            Returns:
                None
        """
        matplotlib.use('TkAgg')
        if self._distances_simulated is None or self._distances_observed is None:
            raise ValueError("The distances have not been calculated yet. Call topical_changes() first!")
        top = self.top_words(chunk=None, number=3, importance=True)
        headers = [f"{k}: {', '.join(top[k])}" for k in range(self._K)]
        similarities_observed = 1 - self._distances_observed[self._warmup + 1:, :]
        similarities_simulated = 1 - self._distances_simulated[self._warmup + 1:, :]
        sns.set(style="darkgrid")
        if len(plot_args) == 0:
            plot_args = {"nrows": math.floor(self._K / 5), "ncols": 5, "figsize": (20, self._K)}
        fig, axs = plt.subplots(**plot_args, sharex='col', sharey='row')
        axs = axs.flatten()
        for i in range(similarities_observed.shape[1]):
            sns.lineplot(x=self.chunk_indices[self._date_column].iloc[self._warmup + 1:], y=similarities_observed[:, i], ax=axs[i], color='#4575b4')
            sns.lineplot(x=self.chunk_indices[self._date_column].iloc[self._warmup + 1:], y=similarities_simulated[:, i], ax=axs[i], color='#db9191')

            higher_points = np.where(similarities_simulated[:, i] > similarities_observed[:, i])[0]
            for point in higher_points:
                axs[i].axvline(x=self.chunk_indices[self._date_column].iloc[self._warmup + 1:].iloc[point], color='red', linestyle='--', alpha=0.5)

            axs[i].set_ylabel("Similarities")
            axs[i].set_title(headers[i])
            axs[i].set_xlabel("Date")
            axs[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.suptitle("Topical Changes", fontsize=16, y=1.02)
        plt.show()

    def _simulate_changes(self, mixture: float = 0.9, reference_period: int = 4, quantile_threshold: float = 0.99,
                          samples: int = 500, save_path: str = None, load_path: str = None) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        """
            Simulates the topic proportions for each chunk and calculates the distances between the observed and simulated topic proportions.
            Args:
                mixture: The mixture parameter as an a-priori-estimator for expected change in word usage from one time-chunk to the next. Lower values
                         will detect more change points, but the changes will be smaller. Higher values will detect fewer change points, but the changes
                         will be more noticeable.
                reference_period: The maximum number of chunks to look back for when calculating the topic proportions in the simulation. This can make
                                  the simulation more robust to outliers and allows for incorporating seasonal effects, e.g. when including the last four
                                  chunks when making a quarterly analysis. When using a high value, the change detection will detect gradual changes, since
                                  it compares the new chunks to very old chunks and when using a low value, the change detection will detect sudden changes,
                                  since it compares the new chunks to very recent chunks.
                quantile_threshold: The bootstrap quantile to compare the observed chunk similarities to, to detect significant changes in word usage.
                samples: The number of samples to draw for the simulation.
                save_path: The path to a directory in which the simulated topic proportions should be saved. If None, the simulated topic proportions are not saved.
                load_path: The path to a directory from which the simulated topic proportions should be loaded. If None, the simulated topic proportions are not loaded.
            Returns:
                The distances between the observed and simulated topic proportions.
        """
        if not isinstance(mixture, float):
            raise TypeError("mixture must be a float!")
        if mixture < 0 or mixture > 1:
            raise ValueError("mixture must be between 0 and 1!")
        if not isinstance(reference_period, int):
            raise TypeError("reference_period must be an integer!")
        if reference_period < 1:
            raise ValueError("reference_period must be a natural number greater than 0")
        if not isinstance(quantile_threshold, float):
            raise TypeError("quantile_threshold must be a float!")
        if quantile_threshold < 0 or quantile_threshold > 1:
            raise ValueError("quantile_threshold must be between 0 and 1!")
        if not isinstance(samples, int):
            raise TypeError("samples must be an integer!")
        if samples < 1:
            raise ValueError("samples must be a natural number greater than 0")
        if not isinstance(save_path, str) and save_path is not None or not isinstance(load_path, str) and load_path is not None:
            raise TypeError("save_path and load_path must be string or None!")
        distances_simulated = np.zeros((len(self.chunk_indices), self._K))
        assignments = np.array([self.get_word_topic_matrix(chunk=chunk).transpose() for chunk, row in self.chunk_indices.iterrows()])
        topics = assignments.transpose((1, 2, 0))

        if load_path is not None:
            self._distances_observed = pd.read_csv(os.path.join(load_path, "observations.csv")).to_numpy()
            for chunk in range(self._warmup + 1, len(self.chunk_indices)):
                chunk_simulated = pd.read_csv(os.path.join(load_path, f"simulation_{chunk}.csv"))
                distances_simulated[chunk, :] = np.quantile(chunk_simulated, quantile_threshold, axis=0)
            self._distances_simulated = distances_simulated
            return

        def calculate_phi(x: np.ndarray) -> float: return (x + self._gamma) / (np.sum(x, axis=1)[:, np.newaxis] + x.shape[1] * self._gamma)
        phi_chunks = np.array([calculate_phi(x) for x in assignments])
        distances_observed = np.zeros((len(phi_chunks), self._K))
        run_length = np.array([min(self._warmup, reference_period) for x in range(self._K)])
        self.runs = [run_length.copy()]
        for i, chunk in enumerate(phi_chunks):
            chunk_simulated = np.zeros((samples, self._K))
            if i <= self._warmup:
                self.runs.append(run_length.copy())
                continue
            lookback_by_topic = [min(run_length[x], reference_period) for x in range(self._K)]
            topic_frequencies = assignments[i].sum(axis=1)
            for topic in range(self._K):
                topics_run = topics[topic, :, max(1, i - lookback_by_topic[topic]):i].sum(axis=1)
                distances_observed[i, topic] = cosine(topics[topic, :, i], topics_run)
                topics_tmp = assignments[max(1, i - lookback_by_topic[topic]):i, :, :].sum(axis=0) + self._gamma
                phi = topics_tmp / np.sum(topics_tmp, axis=1)[:, np.newaxis]
                topics_tmp = assignments[i] + self._gamma
                phi_tmp = phi[topic, :].copy()
                phi = topics_tmp / np.sum(topics_tmp, axis=1)[:, np.newaxis]
                phi_tmp = (1 - mixture) * phi_tmp + mixture * phi[topic, :]
                simulations = self._sample(phi_tmp, topic_frequencies[topic], topics_run, samples)
                chunk_simulated[:, topic] = simulations
                distances_simulated[i, topic] = np.quantile(simulations, quantile_threshold)
            run_length = np.where(distances_simulated[i, :] < distances_observed[i, :], 1, run_length + 1)
            self.runs.append(run_length.copy())
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                pd.DataFrame(chunk_simulated).to_csv(os.path.join(save_path, f"simulation_{i}.csv"), index=False)
        if save_path is not None:
            pd.DataFrame(distances_observed).to_csv(os.path.join(save_path, f"observations.csv"), index=False)
        self._distances_simulated = distances_simulated
        self._distances_observed = distances_observed
        return distances_simulated, distances_observed

    def _sample(self, phi, frequency, topics_run, samples) -> List[float]:
        """
            Samples the topic proportions for the simulation.
            Args:
                phi: The topic proportions of the last chunks.
                frequency: The number of words in the current time chunk assigned to the topic to be simulated.
                topics_run: The topic proportions of the last chunk.
                samples: The number of samples to draw.
            Returns:
                The sampled topic proportions.
        """
        if not isinstance(phi, np.ndarray):
            raise TypeError("phi must be a numpy array!")
        if not isinstance(frequency, int):
            try:
                frequency = int(frequency)
            except ValueError:
                raise TypeError("frequency must be an integer!")
        if frequency < 1:
            raise ValueError("frequency must be a natural number greater than 0")
        if not isinstance(topics_run, np.ndarray):
            raise TypeError("topics_run must be a numpy array!")
        if not isinstance(samples, int):
            raise TypeError("samples must be an integer!")
        if samples < 1:
            raise ValueError("samples must be a natural number greater than 0")
        distances = []
        sample = np.random.choice(len(self.lda.get_vocab()), frequency * samples, p=phi)
        sample = sample.reshape((samples, frequency))
        vocab_size = len(self.lda.get_vocab())
        for i in range(samples):
            unique, tmp_counts = np.unique(sample[i, :], return_counts=True)
            counts = np.zeros(vocab_size)
            counts[unique] = tmp_counts
            distances.append(cosine(counts, topics_run))
        return distances

    def wordclouds(self, chunks: List[int] = None, topic: int = None, number: int = 50, path: str = "wordclouds",
                   height: int = 500, width: int = 700, show: bool = True):
        """
            Plots the wordclouds for the given chunk. If chunk is None, the wordclouds are plotted for every time chunk.
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
            try:
                chunks = list(chunks)
            except ValueError:
                raise TypeError("chunks must be a list or None!")
        if isinstance(chunks, list) and not all([isinstance(x, int) for x in chunks]):
            raise TypeError("chunks must be a list of integers!")
        if not isinstance(topic, int) and topic is not None:
            try:
                topic = int(topic)
            except ValueError:
                raise TypeError("topic must be an integer or None!")
        if not isinstance(number, int):
            try:
                number = int(number)
            except ValueError:
                raise TypeError("number must be an integer!")
        if number < 1:
            raise ValueError("number must be a natural number greater than 0")
        if not isinstance(path, str) and path is not None:
            raise TypeError("path must be a string or None!")
        if not isinstance(width, int):
            try:
                width = int(width)
            except ValueError:
                raise TypeError("width must be an integer!")
        if width < 1:
            raise ValueError("width must be a natural number greater than 0")
        if not isinstance(height, int):
            try:
                height = int(height)
            except ValueError:
                raise TypeError("height must be an integer!")
        if height < 1:
            raise ValueError("height must be a natural number greater than 0")
        if not isinstance(show, bool):
            try:
                show = bool(show)
            except ValueError:
                raise TypeError("show must be a boolean!")

        if chunks is None:
            chunks = range(len(self.chunk_indices))
        for chunk in chunks:
            word_topic_matrix = self.get_word_topic_matrix(chunk)
            if path is not None and not os.path.exists(path):
                os.makedirs(path)
                path += "/" if path[-1] != "/" else ""
            self.lda.wordclouds(topic=topic, number=number, path=f"{path}chunk{chunk}.pdf", height=height, width=width,
                                show=show, word_topic_matrix=word_topic_matrix)
