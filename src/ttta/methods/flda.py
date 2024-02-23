import copy

import numpy as np
import random
from typing import Union, List, Tuple, Callable
from itertools import product
import pandas as pd
from tqdm import tqdm
from .LDA.flda_c import *
from ..preprocessing.preprocess import create_dtm, get_word_and_doc_vector
from scipy.sparse import csr_matrix

class FLDA:
    def __init__(self, data: pd.DataFrame, topics_per_layer: List[int], std_dev_alpha: float = 1, std_dev_alpha_bias: float = 1, std_dev_omega: float = 0.5,
                 std_dev_omega_bias: float = 0.5, local_step_alpha: float = 1e-2, global_step_alpha: float = None, bias_step_alpha: float = None,
                 local_step_omega: float = 1e-3, bias_step_omega: float = None, bias_step: float = 1e-3, sparsity_parameters: List[float] = None,
                 bias_alpha: float = -5, bias_omega: float = -5, likelihood_frequency: int = 100, eval_as_block_every: int = 1, prefix: str = None,
                 verbose: int = 1, min_count: int = 2, max_assign: bool = False, ) -> None:
        self.burned_in = False
        self._verbose = verbose
        self._max_assign = max_assign
        self._min_count = min_count
        if global_step_alpha is None:
            global_step_alpha = local_step_alpha / 100
        if bias_step_alpha is None:
            bias_step_alpha = local_step_alpha / 100
        if bias_step_omega is None:
            bias_step_omega = local_step_omega / 100
        if sparsity_parameters is None:
            sparsity_parameters = [0.1, 0.1]
        self._vocab = []
        self._deleted_indices = None
        self._dtm = np.ndarray()
        self._create_dtm(data["texts"])
        self.length_of_documents = np.sum(self._dtm, axis=1)
        self.number_of_docs, self.number_of_words = self._dtm.shape
        self._word_vec, self._doc_vec = get_word_and_doc_vector(self._dtm)
        self._total_number_of_tokens = len(self._word_vec)
        self.doc_meta_mapping = []
        meta_columns = [col for col in data.columns if "meta" in col]
        if len(meta_columns) > 0:
            self.doc_meta_mapping = data.apply(lambda row: ",".join(row[meta_columns]), axis=1).tonumpy()
            meta_lengths = []
            metas = []
            for col in meta_columns:
                meta_lengths.append(len(data[col].unique()))
                metas.append(data[col].unique())
            metas = [",".join(x) for x in product(*metas)]
            self.topic_meta_mapping = np.array(metas * np.prod(topics_per_layer))
            topics_per_layer.extend(meta_lengths)
            self.document_meta_matrix = np.zeros((self.number_of_docs, np.prod(topics_per_layer)))
            for doc, meta in enumerate(self.doc_meta_mapping):
                self.document_meta_matrix[doc][np.argwhere(self.topic_meta_mapping == meta)] = 1
            self.meta = True
        else:
            self.document_meta_matrix = np.ones((self.number_of_docs, np.prod(topics_per_layer)))
            self.meta = False
        random.seed()
        self.number_of_layers = len(topics_per_layer)  # number of factor layers
        self.topics_per_layer = topics_per_layer  # number of topics per layer
        self.Ksub = np.append(1, np.cumprod(topics_per_layer[::-1])[:-1])[::-1]  # used for mapping all topics within each layer
        self.K = np.prod(topics_per_layer)  # total number of topics

        # Returns the place of the topic in the x-dimensional topic matrix
        self.topic_matrix_location = {}
        for topic in range(self.K):
            layer_location = [0 for _ in range(self.number_of_layers)]
            topic_copy = topic
            for layer in range(self.number_of_layers):
                layer_location[layer] = topic_copy // self.Ksub[layer]
                topic_copy %= self.Ksub[layer]
            self.topic_matrix_location[topic] = layer_location

        # Setting global variables
        self.std_dev_alpha = std_dev_alpha
        self.std_dev_alpha_bias = std_dev_alpha_bias
        self.std_dev_omega = std_dev_omega
        self.std_dev_omega_bias = std_dev_omega_bias
        self.local_step_alpha = local_step_alpha
        self.global_step_alpha = global_step_alpha
        self.bias_step_alpha = bias_step_alpha
        self.local_step_omega = local_step_omega
        self.bias_step_omega = bias_step_omega
        self.bias_step = bias_step
        self.delta = sparsity_parameters
        self.bias_alpha = bias_alpha
        self.bias_omega = bias_omega
        self.likelihood_frequency = likelihood_frequency
        self.eval_as_block_every = eval_as_block_every
        if self.eval_as_block_every == 0:
            self.eval_as_block_every = 1
        self.priorPrefix = prefix  # todo noch abaendern, aktuell nur von data
        self.eta_word = np.zeros(self.number_of_words)
        self.eta_topic_word = [np.zeros((self.topics_per_layer[k], self.number_of_words)) for k in range(self.number_of_layers)]
        if prefix is not None:
            self.read_priors()

        self.alpha_global_tuple_matrix = np.zeros((self.number_of_layers, max(self.topics_per_layer)))
        self.alpha_local_factor_matrices = [np.zeros((k, self.number_of_docs)) for k in self.topics_per_layer]
        self.document_topic_prior = np.zeros((self.number_of_docs, self.K))
        self.omega_word = copy.deepcopy(self.eta_word)
        self.omega_topic_word = copy.deepcopy(self.eta_topic_word)
        self.word_topic_prior = np.zeros((self.K, self.number_of_words))
        self.beta = np.zeros(self.K)

        self.update_priors()
        self.alpha_norm = self.document_topic_prior.sum(axis=1)
        self.omega_norm = self.word_topic_prior.sum(axis=1)

        self.assignments = np.zeros(self._total_number_of_tokens)
        self.all_topics_by_word_in_document = None
        self.document_topic_count = np.zeros((self.number_of_docs, self.K))
        self.topic_word_count = np.zeros((self.K, self.number_of_words))
        self.topic_count = np.zeros(self.K)
        for index in range(self._total_number_of_tokens):
            word = self._word_vec[index]
            doc = self._doc_vec[index]

            # sample initial value from word priors
            sampled_topic = -1
            topic_probabilities = np.zeros(self.K)
            total_probability = 0
            highest_probability = 0
            for topic in range(self.K):
                if self.doc_meta_mapping[doc] == self.topic_meta_mapping[topic]:
                    topic_probabilities[topic] = self.word_topic_prior[topic][word]
                    total_probability += topic_probabilities[topic]
                    if topic_probabilities[topic] > highest_probability:
                        highest_probability = topic_probabilities[topic]
            random_threshold = random.random() * total_probability
            cumsum_of_probabilities = 0.0
            for topic in range(self.K):
                cumsum_of_probabilities += topic_probabilities[topic]
                if cumsum_of_probabilities > random_threshold:
                    sampled_topic = topic
                    break
            self.assignments[index] = sampled_topic

            # update counts
            self.topic_word_count[sampled_topic][word] += 1
            self.topic_count[sampled_topic] += 1
            self.document_topic_count[doc][sampled_topic] += 1

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

    def get_topic_matrix_location(self, x):
        return self.topic_matrix_location[x]

    def vectorToI(self, z):
        return sum(z * self.Ksub)

    def logistic(self, x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    def dlogistic(self, x):
        return self.logistic(x) * (1.0 - self.logistic(x))

    def update_weights(self):
        self.update_weights_omega()
        self.update_weights_alpha()

    def update_weights_alpha(self):
        sigma = self.std_dev_alpha
        gradient_topic = [np.zeros(z) for z in self.topics_per_layer]
        dg = np.vectorize(digamma_func)(self.alpha_norm) - np.vectorize(digamma_func)(self.alpha_norm + self.length_of_documents)
        dgW = np.vectorize(digamma_func)(self.document_topic_count + self.document_topic_prior + 1e-16) - np.vectorize(digamma_func)(self.document_topic_prior + 1e-16)
        total_dg = dg.reshape((len(dg), 1)) + dgW
        gradient_likelihood = self.document_topic_prior * total_dg * self.document_meta_matrix
        gradientB = np.sum(gradient_likelihood)
        gradientBeta = np.sum(gradient_likelihood * (1-self.logistic(self.beta)))
        gradient_document_topic = [np.zeros(z) for z in self.topics_per_layer]
        all_topic_locations = [self.get_topic_matrix_location(x) for x in range(self.K)]  # todo noch letzten gradienten unten optimieren, problem: topic zu topic in matrix bzw umgekehrt
        for layer in range(self.number_of_layers):
            for topic_location_in_layer in range(self.topics_per_layer[layer]):
                gradient_topic += gradient_likelihood[:, [x[layer] == topic_location_in_layer for x in all_topic_locations]].sum()
                gradient_document_topic[layer][topic_location_in_layer[layer]] += gradient_likelihood[:, [x[layer] == topic_location_in_layer for x in all_topic_locations]].sum()
        for layer in range(self.number_of_layers):
            for topic_location_in_layer in range(self.topics_per_layer[layer]):
                self.alpha_local_factor_matrices[layer][topic_location_in_layer] += (gradient_document_topic[layer][topic_location_in_layer] - self.alpha_local_factor_matrices[layer] / (sigma ** 2)).sum(axis=0) * self.local_step_alpha
                self.alpha_global_tuple_matrix[layer][topic_location_in_layer] += (gradient_topic[layer][topic_location_in_layer] - self.alpha_global_tuple_matrix[layer] / (sigma ** 2)) * self.global_step_alpha

        gradientB += -self.bias_alpha / (self.std_dev_alpha_bias ** 2)
        self.bias_alpha += self.bias_step_alpha * gradientB
        gradientBeta += (self.delta[0]-1) * self.dlogistic(self.beta) / self.logistic(self.beta) - (self.delta[1]-1) * self.dlogistic(self.beta) / (1-self.logistic(self.beta))
        self.beta += self.bias_step * gradientBeta

    def update_weights_omega(self):
        """
        Gradient descent on all omega
        """
        sigma = self.std_dev_omega
        gradientB = 0

        for word in range(self.number_of_words):
            gradient_word = 0
            gradient_topic_word = [np.zeros(z) for z in self.topics_per_layer]

            for topic in range(self.K):
                topic_location = self.get_topic_matrix_location(topic)
                dg1 = digamma_func(self.omega_norm[topic])
                dg2 = digamma_func(self.omega_norm[topic] + self.topic_count[topic])
                dgW1 = digamma_func(self.word_topic_prior[topic][word] + self.topic_word_count[topic][word])
                dgW2 = digamma_func(self.word_topic_prior[topic][word])

                gradient_likelihood = self.word_topic_prior[topic][word] * (dg1 - dg2 + dgW1 - dgW2)
                for layer in range(self.number_of_layers):
                    gradient_topic_word[layer][topic_location[layer]] += gradient_likelihood

                gradient_word += gradient_likelihood
                gradientB += gradient_likelihood

            for layer in range(self.number_of_layers):
                for topic_location_in_layer in range(self.topics_per_layer[layer]):
                    gradient_topic_word[layer][topic_location_in_layer] += -(self.omega_topic_word[layer][topic_location_in_layer][word] -
                                                                             self.eta_topic_word[layer][topic_location_in_layer][word]) / (sigma ** 2)
                    self.omega_topic_word[layer][topic_location_in_layer][word] += self.local_step_omega * gradient_topic_word[layer][topic_location_in_layer]
            gradient_word += -(self.omega_word[word] - self.eta_word[word]) / (sigma ** 2)
            self.omega_word[word] += self.local_step_omega * gradient_word
        gradientB -= self.bias_omega / (self.std_dev_omega_bias ** 2)
        self.bias_omega += self.bias_step_omega * gradientB

    def do_sampling(self, iter):
        for index in range(self._doc_vec):
            if self.meta:
                self.sample_with_meta_data(index)
            elif self.eval_as_block_every > 0 and iter % self.eval_as_block_every == 0:
                self.sample(index)
            else:
                self.sample_ind(index)

        # 1 iteration of gradient ascent (change 1 to x to do x iterations)
        for i in range(1):
            self.update_weights()

        # Compute the priors with the new params and update the cached prior variables
        self.update_priors()
        self.alpha_norm = self.document_topic_prior.sum(axis=1)
        self.omega_norm = self.word_topic_prior.sum(axis=1)

        if iter % self.likelihood_frequency == 0:
           print("Log-likelihood:", self.log_likelihood())

        # Collect samples (all_topics_by_word_in_document)
        # if self.burned_in:  # todo was mit burn in machen
        #     for index in range(self._total_number_of_tokens):
        #         word_topic = self.assignments[index]
        #         self.all_topics_by_word_in_document[index][word_topic] += 1  # todo all_... abaendern

    def update_priors(self):
        for topic in range(self.K):
            z = self.get_topic_matrix_location(topic)
            b = self.logistic(self.beta[topic])
            omega_weight = self.omega_word + self.bias_omega
            alpha_weight = self.bias_alpha + self.alpha_global_tuple_matrix[[i for i in range(self.number_of_layers)], z].sum()  # todo alpha weights nicht updaten wenn falsche Metadaten
            all_docs_weight = np.full(self.number_of_docs, alpha_weight)
            for layer in range(self.number_of_layers):
                all_docs_weight += self.alpha_local_factor_matrices[layer][z[layer]]
                omega_weight += self.omega_topic_word[layer][topic[layer]]
            self.document_topic_prior[:, topic] = b * np.exp(all_docs_weight)
            self.word_topic_prior[topic] = np.exp(omega_weight)
        # to anti-update alpha for the wrong meta_data
        self.document_topic_prior *= self.document_meta_matrix

    def sample_with_meta_data(self, index):
        word = self._word_vec[index]
        doc = self._doc_vec[index]
        meta = self.doc_meta_mapping[doc]
        sampled_topic = self.assignments[index]

        # Decrement counts
        self.topic_word_count[sampled_topic][word] -= 1
        self.topic_count[sampled_topic] -= 1
        self.document_topic_count[doc][sampled_topic] -= 1


        # Sample new tuple value
        topic_probabilities = np.zeros(self.K)
        total_probability = 0

        for topic in range(self.K):
            if self.topic_meta_mapping[topic] == meta:
                topic_probabilities[topic] = (self.document_topic_count[doc][topic] + self.document_topic_prior[doc][topic]) * \
                       (self.topic_word_count[topic][word] + self.word_topic_prior[topic][word]) / (self.topic_count[topic] + self.omega_norm[topic])
                total_probability += topic_probabilities[topic]

        random_threshold = np.random.rand() * total_probability

        cumsum_of_probabilities = 0.0
        for topic in range(self.K):
            cumsum_of_probabilities += topic_probabilities[topic]
            if cumsum_of_probabilities > random_threshold:
                sampled_topic = topic
                break

        # Increment counts
        self.topic_word_count[sampled_topic][word] += 1
        self.topic_count[sampled_topic] += 1
        self.document_topic_count[doc][sampled_topic] += 1

        # Set new assignments
        self.assignments[index] = sampled_topic

    def sample(self, index):
        word = self._word_vec[index]
        doc = self._doc_vec[index]
        sampled_topic = self.assignments[index]

        # Decrement counts
        self.topic_word_count[sampled_topic][word] -= 1
        self.topic_count[sampled_topic] -= 1
        self.document_topic_count[doc][sampled_topic] -= 1

        # Sample new tuple value
        topic_probabilities = np.zeros(self.K)
        total_probability = 0

        for topic in range(self.K):
            topic_probabilities[topic] = (self.document_topic_count[doc][topic] + self.document_topic_prior[doc][topic]) * \
                   (self.topic_word_count[topic][word] + self.word_topic_prior[topic][word]) / (self.topic_count[topic] + self.omega_norm[topic])

            total_probability += topic_probabilities[topic]

        random_threshold = np.random.rand() * total_probability

        cumsum_of_probabilities = 0.0
        for topic in range(self.K):
            cumsum_of_probabilities += topic_probabilities[topic]
            if cumsum_of_probabilities > random_threshold:
                sampled_topic = topic
                break

        # Increment counts
        self.topic_word_count[sampled_topic][word] += 1
        self.topic_count[sampled_topic] += 1
        self.document_topic_count[doc][sampled_topic] += 1

        # Set new assignments
        self.assignments[index] = sampled_topic

    def sample_ind(self, index):
        word = self._word_vec[index]
        doc = self._doc_vec[index]
        sampled_topic = self.assignments[index]

        # Decrement counts
        self.topic_word_count[sampled_topic][word] -= 1
        self.topic_count[sampled_topic] -= 1
        self.document_topic_count[doc][sampled_topic] -= 1

        topic_location = self.get_topic_matrix_location(sampled_topic)
        new_topic_location = np.zeros(self.number_of_layers, dtype=int)

        # Sample new value for each factor
        for layer in range(self.number_of_layers):
            topic_probabilities = np.zeros(self.topics_per_layer[layer])
            total_probability = 0

            old_topic_location = topic_location.copy()

            for topic_location_in_layer in range(self.topics_per_layer[layer]):
                old_topic_location[layer] = topic_location_in_layer
                x = self.get_topic_matrix_location(old_topic_location)
                topic_probabilities[topic_location_in_layer] = (self.document_topic_count[doc][x] + self.document_topic_prior[doc][x]) * \
                                                               (self.topic_word_count[x][word] + self.word_topic_prior[x][word]) / (self.topic_count[x] + self.omega_norm[x])

                total_probability += topic_probabilities[topic_location_in_layer]

            random_threshold = np.random.rand() * total_probability

            cumsum_probabilities = 0.0
            for topic_location_in_layer in range(self.topics_per_layer[layer]):
                cumsum_probabilities += topic_probabilities[topic_location_in_layer]
                if cumsum_probabilities > random_threshold:
                    new_topic_location[layer] = topic_location_in_layer
                    break

        sampled_topic = self.get_topic_matrix_location(new_topic_location)

        # Increment counts
        self.topic_word_count[sampled_topic][word] += 1
        self.topic_count[sampled_topic] += 1
        self.document_topic_count[doc][sampled_topic] += 1

        # Set new assignments
        self.assignments[index] = sampled_topic

    def run(self, iterations, samples, filename):  # todo noch nicht ueberprueft
        for i in tqdm(range(iterations)):
            if i > (iterations - samples):
                self.burned_in = True
            self.do_sampling(i)
        self.write_output(filename)

    def log_likelihood(self):
        likelihood = np.log(np.matmul((self.document_topic_count + self.document_topic_prior) /
                                      (self.length_of_documents + self.alpha_norm).reshape((len(self.length_of_documents), 1)),
                                      (self.topic_word_count + self.word_topic_prior) /
                                      (self.topic_count + self.omega_norm).reshape((len(self.topic_count), 1))))
        likelihood = sum([likelihood[x, y] for x, y in zip(self._doc_vec, self._word_vec)])
        return likelihood

    def read_priors(self):
        with open(f"{self.priorPrefix}.txt", 'r') as file:
            for line in file:
                tokens = line.split()
                word = tokens[0]
                if word not in self.word_map:
                    continue
                word = self.word_map[word]
                weight = float(tokens[1])
                self.eta_word[word] = weight

        for layer in range(self.number_of_layers):
            for topic_location_in_layer in range(self.topics_per_layer[layer]):
                with open(f"{self.priorPrefix}{layer}_{topic_location_in_layer}.txt", 'r') as file:
                    for line in file:
                        tokens = line.split()
                        word = tokens[0]
                        if word not in self.word_map:
                            continue
                        word = self.word_map[word]
                        weight = float(tokens[1])
                        self.eta_topic_word[layer][topic_location_in_layer][word] = weight

    def write_output(self, filename):
        with open(f"{filename}.assign", 'w') as file:
            for doc in range(self.number_of_docs):

                for word_location in range(self.length_of_documents[doc]):
                    word = self.word_map_inv[self.docs[doc][word_location]]
                    file.write(word)
                    for topic in range(self.K):
                        file.write(f":{self.all_topics_by_word_in_document[doc][word_location][topic]}")
                    file.write(" ")

                file.write("\n")

        for i in range(self.number_of_layers):
            with open(f"{filename}.omegaZW{i}", 'w') as file:
                for w in range(self.number_of_words):
                    word = self.word_map_inv[w]
                    file.write(word)
                    for z in range(self.topics_per_layer[i]):
                        file.write(f" {self.omega_topic_word[i][z][w]}")
                    file.write("\n")

        with open(f"{filename}.omegaW", 'w') as file:
            for w in range(self.number_of_words):
                word = self.word_map_inv[w]
                file.write(f"{word} {self.omega_word[w]}\n")

        with open(f"{filename}.omegaB", 'w') as file:
            file.write(f"{self.bias_omega}\n")

        for i in range(self.number_of_layers):
            with open(f"{filename}.alphaZ{i}", 'w') as file:
                for z in range(self.topics_per_layer[i]):
                    file.write(f"{self.alpha_global_tuple_matrix[i][z]}\n")

        with open(f"{filename}.alphaB", 'w') as file:
            file.write(f"{self.bias_alpha}\n")

        with open(f"{filename}.beta", 'w') as file:
            for x in range(self.K):
                file.write(f"{self.logistic(self.beta[x])}\n")
