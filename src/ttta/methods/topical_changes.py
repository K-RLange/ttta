"""This module contains a class to detect topical changes in a diachronic topic model."""
from typing import Union, List, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib
from .rolling_lda import RollingLDA
import seaborn as sns
from matplotlib import pyplot as plt
import math
import pickle
import os
from scipy.spatial.distance import cosine
from operator import itemgetter

class TopicalChanges:
    """This class is used to detect topical changes in a diachronic topic model."""

    def __init__(self, roll: RollingLDA, mixture: float = 0.9, reference_period: int = 4, quantile_threshold: float = 0.99, samples: int = 500, save_path: str = None,
                 load_path: str = None, fast: bool = True):
        """Calculates topical change points for the given parameters.

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
            fast: Whether the word impact should be calculated fast. If True, words that do not occur in the compared time chunks are ignored during
                  the word impact calculation. This should not change the results, unless "word_impact_number" is set to a high value compared to the
                  vocabulary size.
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
                if reference_period == int(reference_period):
                    reference_period = int(reference_period)
                else:
                    raise ValueError
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
                if samples == int(samples):
                    samples = int(samples)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("samples must be an integer!")
        if samples < 1:
            raise ValueError("samples must be a natural number greater than 0!")
        if not isinstance(fast, bool):
            raise TypeError("fast must be a boolean!")
        if not isinstance(roll, RollingLDA):
            raise TypeError("roll must be a RollingLDA object!")

        self.mixture = mixture
        self._roll = roll
        self.reference_period = reference_period
        self.quantile_threshold = quantile_threshold
        self.samples = samples
        self.fast = fast
        self._simulate_changes(save_path, load_path)


    def plot_distances(self, show: bool = True, **plot_args) -> None:
        """Plots the distances between the observed and simulated topic proportions.

        Args:
            show: Whether to show the plot or not.
            **plot_args: Additional arguments to be passed to the matplotlib.pyplot.subplot function, e.g. nrows, ncols and figsize.
        Returns:
            None
        """
        if len([np.argwhere(self._distances_simulated.transpose()[x] < self._distances_observed.transpose()[x]).transpose(1, 0)[0].tolist() for x in
                          range(len(self._distances_simulated.transpose()))]) < 1:
            raise ValueError("No change points detected!")
        matplotlib.use('TkAgg')
        top = self._roll.top_words(chunk=None, number=3, importance=True, return_as_data_frame=False)
        headers = [f"{k}: {', '.join(top[k])}" for k in range(self._roll._K)]
        similarities_observed = 1 - self._distances_observed[self._roll._warmup + 1:, :]
        similarities_simulated = 1 - self._distances_simulated[self._roll._warmup + 1:, :]
        sns.set(style="darkgrid")
        if len(plot_args) == 0:
            plot_args = {"nrows": max(1, math.floor(self._roll._K / 5)), "ncols": 5, "figsize": (20, self._roll._K)}
        fig, axs = plt.subplots(**plot_args, sharex='col', sharey='row')
        axs = axs.flatten()
        for i in range(similarities_observed.shape[1]):
            sns.lineplot(x=self._roll.chunk_indices[self._roll._date_column].iloc[self._roll._warmup + 1:], y=similarities_observed[:, i], ax=axs[i], color='#4575b4')
            sns.lineplot(x=self._roll.chunk_indices[self._roll._date_column].iloc[self._roll._warmup + 1:], y=similarities_simulated[:, i], ax=axs[i], color='#db9191')

            higher_points = np.where(similarities_simulated[:, i] > similarities_observed[:, i])[0]
            for point in higher_points:
                axs[i].axvline(x=self._roll.chunk_indices[self._roll._date_column].iloc[self._roll._warmup + 1:].iloc[point], color='red', linestyle='--', alpha=0.5)

            axs[i].set_ylabel("Similarities")
            axs[i].set_title(headers[i])
            axs[i].set_xlabel("Date")
            axs[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.suptitle("Topical Changes", fontsize=16, y=1.02)
        if show:
            plt.show()

    def word_impact(self, number: int = 5, date_format: str = "%Y-%m-%d", fast: bool = True) -> pd.DataFrame:
        """Calculate the leave one out word impact for each topic and change point.

        Args:
            number: The number of words to return for the word impact.
            date_format: The date format to return alongside the changes.
            fast: Whether the word impact should be calculated fast. If True, words that do not occur in the compared time chunks are ignored during
                  the word impact calculation. This should not change the results, unless "number" is set to a high value compared to the
                  vocabulary size.
        Returns:
            A pandas DataFrame containing the date and word impact for each topic and change point.
        """
        if not isinstance(number, int):
            try:
                if number == int(number):
                    number = int(number)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("number must be an integer!")
        if number < 1:
            raise ValueError("number must be a natural number greater than 0")
        if not isinstance(date_format, str):
            raise TypeError("date_format must be a string!")
        if not isinstance(fast, bool):
            raise TypeError("fast must be a boolean!")

        assignments = np.array([self._roll.get_word_topic_matrix(chunk=chunk).transpose() for chunk, row in self._roll.chunk_indices.iterrows()])
        topics = assignments.transpose((1, 2, 0))
        change_indices = [np.argwhere(self._distances_simulated.transpose()[x] < self._distances_observed.transpose()[x]).transpose(1, 0)[0].tolist() for x in
                          range(len(self._distances_simulated.transpose()))]
        events_end_date = [[self._roll.chunk_indices[self._roll._date_column].iloc[y] for y in x] for x in change_indices]
        leave_one_out_word_impact = {"Topic": [], "Date": [], "Significant Words": []}
        for k, changes in enumerate(change_indices):
            if len(changes) == 0:
                continue
            for i, change in enumerate(changes):
                temp = topics[k, :, max(1, change - self.runs[change][k], change - self.reference_period):change].sum(axis=1)
                loo_temp = cosine(topics[k, :, change], temp)
                leave_one_out_substracted = np.array(
                    [0 if fast and topics[k, word, change] == 0 and temp[
                        word] == 0 else cosine(
                        np.delete(topics[k, :, change], word),
                        np.delete(temp, word)) - loo_temp for word in
                     range(topics.shape[1])])
                significant_words = itemgetter(
                    *leave_one_out_substracted.argsort()[:number].astype(
                        np.uint64).tolist())(self._roll.lda.get_vocab())
                leave_one_out_word_impact["Topic"].append(k)
                leave_one_out_word_impact["Date"].append(events_end_date[k][i].strftime(date_format))
                leave_one_out_word_impact["Significant Words"].append(significant_words)
        leave_one_out_word_impact = pd.DataFrame(leave_one_out_word_impact)
        return leave_one_out_word_impact

    def _simulate_changes(self, save_path: str = None, load_path: str = None) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        """Simulate the topic proportions for each chunk and calculates the distances between the observed and simulated topic proportions.

        Args:
            save_path: The path to a directory in which the simulated topic proportions should be saved. If None, the simulated topic proportions are not saved.
            load_path: The path to a directory from which the simulated topic proportions should be loaded. If None, the simulated topic proportions are not loaded.
        Returns:
            The distances between the observed and simulated topic proportions.
        """
        if (not isinstance(save_path, str) and save_path is not None) or (not isinstance(load_path, str) and load_path is not None):
            raise TypeError("save_path and load_path must be string or None!")
        distances_simulated = np.zeros((len(self._roll.chunk_indices), self._roll._K))
        assignments = np.array([self._roll.get_word_topic_matrix(chunk=chunk).transpose() for chunk, row in self._roll.chunk_indices.iterrows()])
        topics = assignments.transpose((1, 2, 0))

        if load_path is not None:
            self._distances_observed = pd.read_csv(os.path.join(load_path, "observations.csv")).to_numpy()
            for chunk in range(self._roll._warmup + 1, len(self._roll.chunk_indices)):
                chunk_simulated = pd.read_csv(os.path.join(load_path, f"simulation_{chunk}.csv"))
                distances_simulated[chunk, :] = np.quantile(chunk_simulated, self.quantile_threshold, axis=0)
            self._distances_simulated = distances_simulated
            return

        def calculate_phi(x: np.ndarray) -> float: return (x + self._roll._gamma) / (np.sum(x, axis=1)[:, np.newaxis] + x.shape[1] * self._roll._gamma)
        phi_chunks = np.array([calculate_phi(x) for x in assignments])
        distances_observed = np.zeros((len(phi_chunks), self._roll._K))
        run_length = np.array([min(self._roll._warmup, self.reference_period) for x in range(self._roll._K)])
        self.runs = [run_length.copy()]
        for i, chunk in enumerate(phi_chunks):
            chunk_simulated = np.zeros((self.samples, self._roll._K))
            if i <= self._roll._warmup:
                self.runs.append(run_length.copy())
                continue
            lookback_by_topic = [min(run_length[x], self.reference_period) for x in range(self._roll._K)]
            topic_frequencies = assignments[i].sum(axis=1)
            for topic in range(self._roll._K):
                topics_run = topics[topic, :, max(1, i - lookback_by_topic[topic]):i].sum(axis=1)
                distances_observed[i, topic] = cosine(topics[topic, :, i], topics_run)
                topics_tmp = assignments[max(1, i - lookback_by_topic[topic]):i, :, :].sum(axis=0) + self._roll._gamma
                phi = topics_tmp / np.sum(topics_tmp, axis=1)[:, np.newaxis]
                topics_tmp = assignments[i] + self._roll._gamma
                phi_tmp = phi[topic, :].copy()
                phi = topics_tmp / np.sum(topics_tmp, axis=1)[:, np.newaxis]
                phi_tmp = (1 - self.mixture) * phi_tmp + self.mixture * phi[topic, :]
                simulations = self._sample(phi_tmp, topic_frequencies[topic], topics_run)
                chunk_simulated[:, topic] = simulations
                distances_simulated[i, topic] = np.quantile(simulations, self.quantile_threshold)
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

    def _sample(self, phi, frequency, topics_run) -> List[float]:
        """Sample the topic proportions for the simulation.

        Args:
            phi: The topic proportions of the last chunks.
            frequency: The number of words in the current time chunk assigned to the topic to be simulated.
            topics_run: The topic proportions of the last chunk.
        Returns:
            The sampled topic proportions.
        """
        if not isinstance(phi, np.ndarray):
            raise TypeError("phi must be a numpy array!")
        if not isinstance(frequency, int):
            try:
                if frequency == int(frequency):
                    frequency = int(frequency)
                else:
                    raise ValueError
            except ValueError:
                raise TypeError("frequency must be an integer!")
        if frequency < 1:
            raise ValueError("frequency must be a natural number greater than 0")
        if not isinstance(topics_run, np.ndarray):
            raise TypeError("topics_run must be a numpy array!")
        distances = []
        sample = np.random.choice(len(self._roll.lda.get_vocab()), frequency * self.samples, p=phi)
        sample = sample.reshape((self.samples, frequency))
        vocab_size = len(self._roll.lda.get_vocab())
        for i in range(self.samples):
            unique, tmp_counts = np.unique(sample[i, :], return_counts=True)
            counts = np.zeros(vocab_size)
            counts[unique] = tmp_counts
            distances.append(cosine(counts, topics_run))
        return distances

    def save(self, path: str) -> None:
        """
        Save the model to a pickle file.

        Returns:
            None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        pickle.dump(self, open(path, "wb"))

    def load(self, path: str) -> None:
        """
        Load a model from a pickle file.

        Returns:
            None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        loaded = pickle.load(open(path, "rb"))
        self.__dict__ = loaded.__dict__