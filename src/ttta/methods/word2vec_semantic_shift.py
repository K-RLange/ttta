import pickle
import warnings
import copy
from nltk.tag import pos_tag
from typing import List, Union, Optional, Tuple, Literal
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from ttta.preprocessing.chunk_creation import _get_time_indices
from gensim.models.phrases import Phrases, Phraser
from datetime import datetime

class Word2VecSemanticShift:
    """Word2VecSemanticShift class for training and aligning Word2Vec models
    for semantic shift analysis.

    Methods:
        fit: Fits the Word2Vec models on the texts.
        fit_update: Updates the Word2Vec models on the texts.
        infer_vector: Infers the vector of a word.
        top_words: Returns the top words similar to a word.
        visualize: Visualizes the semantic shift of a word across chunks.
        get_vector: Returns the vector of a word in a chunk.
        get_parameters: Returns the parameters of the RollingLDA model.
        get_reference: Returns the reference chunk index.
        get_vocab: Returns the vocabulary of a chunk.
        is_trained: Returns whether the model has been trained or not.
        save: Saves this class to a file.
        load: Loads this class from a file.
        _prepare_for_training: Prepares the texts for training.
        _train_word2vec: Trainer function for the Word2Vec model.
        _align_models: Aligns the Word2Vec models.
        _check_inference_requirements: Checks the requirements for inference.
    """
    def __init__(
            self,
            how: Union[str, List[datetime]] = "ME",
            min_count: int = 2,
            min_docs_per_chunk: int = 2,
            window: int = 5,
            negative: int = 5,
            ns_exponent: float = 0.75,
            vector_size: int = 300,
            alpha: float = 0.025,
            max_vocab_size: Optional[int] = None,
            sample: float = 1e-3, 
            seed: int = 1,
            workers: int = 3,
            min_alpha: float = 0.0001,
            sg: int = 0,
            hs: int = 0,
            cbow_mean: int = 1,
            null_word: int = 0,
            trim_rule: Optional[str] = None,
            sorted_vocab: int = 1,
            compute_loss: bool = False,
            callbacks: Optional[List] = None,
            comment: Optional[str] = None,
            max_final_vocab: Optional[int] = None, 
            shrink_windows: bool = False,
            bigram_min_count: int = 2,
            bigram_threshold: float = 1.0,
            verbose: int = 1,
            ):
        """
        Args:
        how: List of datetime dates indicating the end of time chunks or a string indicating the frequency of the time chunks as in pandas.resample().
        min_count: Ignores all words with total frequency lower than this.
        min_docs_per_chunk: The minimum number of documents a chunk must contain to be used for the training.
        window: The maximum distance between the current and predicted word within a sentence.
        negative: If > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn. If set to 0, no negative sampling is used.
        ns_exponent: The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
        vector_size: Dimensionality of the word vectors.
        alpha: The initial learning rate.
        max_vocab_size: Limits the RAM during vocabulary building. If there are more unique words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM. Set to None for no limit.
        sample: The threshold for configuring which higher-frequency words are randomly downsampled. Highly influencial.
        seed: Seed for the random number generator.
        workers: Use these many worker threads to train the model.
        min_alpha: Learning rate will linearly drop to min_alpha as training progresses.
        sg: Training algorithm: 1 for skip-gram; otherwise CBOW.
        hs: If 1, hierarchical softmax will be used for model training. If set to 0, and negative is non-zero, negative sampling will be used.
        cbow_mean: If 0, use the sum of the context word vectors. If 1, use the mean. Only applies when cbow is used.
        null_word: If > 0, the model uses the null word to fit the model to the training data. If 0, no null word is used.
        trim_rule: Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary, be trimmed away, or handled using the default. Can be None (min_count will be used), or a string that specifies the rule.
        sorted_vocab: If 1, sort the vocabulary by descending frequency before assigning word indexes.
        compute_loss: If True, computes and stores loss value which can be retrieved using get_latest_training_loss().
        callbacks: Sequence of callbacks to be executed at specific stages during training.
        comment: A comment to be added to the model.
        max_final_vocab: Limits the vocabulary size when building the vocabulary. If there are more unique words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM. Set to None for no limit.
        shrink_windows: Whether to shrink the window size as training progresses.
        bigram_min_count: Ignore all words and bigrams with total collected count lower than this value.
        bigram_threshold: Represent a score threshold for forming the phrases (higher means fewer phrases). A phrase of words a and b is accepted if the score of the phrase is greater than threshold.
        verbose: The verbosity level. 0 does not print anything, 1 prints the current progress, 2 prints additional debug information.
        """
        self.trainer_args = {
            "min_count":min_count,
            "window":window,
            "negative":negative,
            "ns_exponent":ns_exponent,
            "vector_size":vector_size,
            "alpha":alpha,
            "max_vocab_size":max_vocab_size,
            "sample":sample,
            "seed":seed,
            "workers":workers,
            "min_alpha":min_alpha,
            "sg":sg,
            "hs":hs,
            "cbow_mean":cbow_mean,
            "null_word":null_word,
            "trim_rule":trim_rule,
            "sorted_vocab":sorted_vocab,
            "compute_loss":compute_loss,
            "callbacks":callbacks,
            "comment":comment,
            "max_final_vocab":max_final_vocab,
            "shrink_windows":shrink_windows
        }
        self._reference = None
        self.how = how
        self.word2vecs: List[Word2Vec] = []
        self.aligned_models: List[Word2Vec] = []
        self._verbose = verbose
        self.chunk_indices = None
        self.sorting = None
        self.min_docs_per_chunk = min_docs_per_chunk
        self.bigram_min_count = bigram_min_count
        self.bigram_threshold = bigram_threshold
        self._last_text = 0

    def fit(
            self,
            texts: pd.DataFrame,
            text_column: str = "text",
            date_column: str = "date",
            date_format: str = None,
            align_reference: int = -1,
            epochs: int = 5,
            start_alpha: float = 0.025,
            end_alpha: float = 0.0001,
        ) -> None:
        """
        Fits the Word2Vec models on the texts.
        Args:
            texts: The texts to train the Word2Vec models on.
            text_column: The column containing the texts.
            date_column: The column containing the dates.
            date_format: The format of the dates.
            align_reference: The reference chunk index to align the models.
            epochs: The number of epochs to train the models.
            start_alpha: The initial learning rate.
            end_alpha: The final learning rate.

        Examples:
            >>> ss = Word2VecSemanticShift()
            >>> data = pd.read_csv("data.csv")
            >>> ss.fit(data, text_column="text", date_column="date", date_format="%Y-%m-%d", align_reference=-1, epochs=5, start_alpha=0.025, end_alpha=0.0001, date_groupby="year")
        """

        self._prepare_for_training(texts, text_column, date_column, date_format)
        if self.is_trained():
            raise ValueError("The model has already been trained. Call 'fit_update' to update the model.")

        iterator = self.chunk_indices.iterrows()
        if self._verbose > 0:
            iterator = tqdm(iterator, total=len(self.chunk_indices), unit="chunk")
        for i, row in enumerate(iterator):
            if self._verbose > 0 and i < len(self.chunk_indices) - 1:
                iterator.set_description(f"Chunk {i + 1}/{len(self.chunk_indices)}")
            end = len(texts) if i + 1 >= len(self.chunk_indices) else int(self.chunk_indices.iloc[i + 1]["chunk_start"])
            model = self._train_word2vec(texts[text_column].iloc[self.chunk_indices["chunk_start"].iloc[i]:end], epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha)
            self.word2vecs.append(model)


        self._reference = align_reference
        self._align_models(self.word2vecs, reference=align_reference)
        self._last_text = len(texts)

    def fit_update(
            self,
            texts: pd.DataFrame,
            text_column: str = "text",
            date_column: str = "date",
            date_format: str = None,
            align_reference: int = -1,
            epochs: int = 5,
            start_alpha: float = 0.025,
            end_alpha: float = 0.0001
        ) -> None:
        """
        Updates the Word2Vec models on the texts.
        Args:
            texts: The texts to train the Word2Vec models on.
            text_column: The column containing the texts.
            date_column: The column containing the dates.
            align_reference: The reference chunk index to align the models.
            epochs: The number of epochs to train the models.
            start_alpha: The initial learning rate.
            end_alpha: The final learning rate.
        """
        if self.word2vecs is None or not self.is_trained():
            raise ValueError("The model has not been trained. Call 'fit' to train the model first.")

        last_trained = len(self.chunk_indices)
        self._prepare_for_training(texts, text_column, date_column, date_format, update=True)
        iterator = self.chunk_indices.iloc[last_trained:].iterrows()
        if self._verbose > 0:
            iterator = tqdm(iterator, unit="chunk")
        for i, row in iterator:
            if self._verbose and i < len(self.chunk_indices) - 1:
                iterator.set_description(
                    f"Processing {self.chunk_indices.iloc[i + 1]['chunk_start'] - 1 - self.chunk_indices.iloc[i]['chunk_start']} documents in "
                    f"chunk {self.chunk_indices.iloc[i][self._date_column].strftime('%Y-%m-%d')}")
            end = self.chunk_indices.iloc[i + 1]["chunk_start"] if i + 1 < len(
                self.chunk_indices) else (len(texts) +
                                          self._last_text + 1)
            model = self._train_word2vec(texts[text_column].iloc[
                                         int(self.chunk_indices[
                                              "chunk_start"].iloc[
                                              i] - self._last_text):int(
                                                     end - self._last_text)],
                                         epochs=epochs,
                                         start_alpha=start_alpha,
                                         end_alpha=end_alpha)
            self.word2vecs.append(model)

        self._reference = align_reference
        self._align_models(self.word2vecs, reference=align_reference)
        self._last_text += len(texts)

    def is_trained(self) -> bool:
        """Returns whether the model has been trained or not.

        Returns:
            A boolean value indicating whether the model has been trained or not.
        """
        return len(self.word2vecs) > 0

    def get_vector(self, word, chunk_index, aligned=True):
        """Returns the vector of a word in a chunk.

        Args:
            word: The word to get the vector of.
            chunk_index: The index of the chunk.
            aligned: Whether to get the vector from the aligned models or not.
        Returns:
            The vector of the word.
        """
        self._check_inference_requirements(aligned)
        if aligned:
            return self.aligned_models[chunk_index].wv.get_vector(word)
        else:
            return self.word2vecs[chunk_index].wv.get_vector(word)

    def get_reference(self) -> int:
        """Returns the reference chunk index.

        Returns:
            The reference chunk index.
        """
        return self._reference

    def get_parameters(self) -> dict:
        """Returns the parameters of the Word2Vec models.

        Returns:
            A dictionary containing the parameters of the Word2Vec models.
        """
        return self.__dict__.copy()

    def get_vocab(self, chunk: Union[int, str] = "reference") -> List[str]:
        """Returns the vocabulary of a chunk.

        Args:
            chunk: The index of the chunk or 'reference'.
        Returns:
            The vocabulary of the chunk.
        """
        if chunk == "reference":
            return self.word2vecs[self._reference].wv.index_to_key
        elif isinstance(chunk, int):
            return self.word2vecs[chunk].wv.index_to_key
        else:
            raise ValueError("chunk must be an integer or 'reference'.")

    def infer_vector(self, word: str, chunk_index: int, norm: bool = False,
                     aligned: bool = True) -> np.ndarray:
        """Infers the vector of a word.

        Args:
            word: The word to infer the vector.
            chunk_index: The index of the chunk.
            norm: Whether to normalize the vector or not.
            aligned: Whether to infer the vector from the aligned models or not.
        Returns:
            The inferred vector of the word.
        """
        self._check_inference_requirements(aligned)
        return self.aligned_models[chunk_index].wv.get_vector(word, norm=norm)

    def top_words(self, word: str, chuck_index: int, k: int = 10, pos_tag: Union[bool, str, List[str]] = False, aligned: bool = True) -> Tuple[List[str], List[float]]:
        """Returns the top words similar to a word.

        Args:
            word: The word to get the top words.
            chuck_index: The index of the chunk.
            k: The number of top words to return.
            pos_tag: The part-of-speech tag of the words to return.
            aligned: Whether to get the top words from the aligned models or not.
        Returns:
            A tuple containing the top words and their similarities.
        """
        self._check_inference_requirements(aligned)
        if aligned:
            return self._get_top_k_words(self.aligned_models[chuck_index],
                                         word, k=k, pos_tagging=pos_tag)
    
        else:
            return self._get_top_k_words(self.word2vecs[chuck_index], word,
                                         k=k, pos_tagging=pos_tag)

    def visualize(
            self, 
            main_word: str, 
            chunks_tocompare: Optional[List[int]] = None,
            reference: Optional[int] = -1,
            k: int = 10, 
            pos_tag: Union[bool, str, List[str]] = False, 
            extra_words: Optional[List[str]] = None, 
            ignore_words: Optional[List[str]] = None,
            aligned: bool = True,
            tsne_perplexity: int = 30,
            tsne_metric: str = 'euclidean',
            tsne_learning_rate: Union[str, int] = 'auto'
            ) -> None:
        """Visualizes the semantic shift of a word across chunks.

        Args:
            main_word: The main word to visualize.
            chunks_tocompare: The chunks to compare. If None, all chunks will be compared.
            reference: The reference chunk index.
            k: The number of top words to return.
            pos_tag: The part-of-speech tag of the words to return.
            extra_words: The extra words to include in the visualization.
            ignore_words: The words to ignore in the visualization.
            aligned: Whether to visualize the aligned models or not.
            tsne_perplexity: The perplexity of the t-SNE algorithm.
            tsne_metric: The metric of the t-SNE algorithm.
            tsne_learning_rate: The learning rate of the t-SNE algorithm.
        """
        self._check_inference_requirements(aligned)
        plot_vocab = extra_words if extra_words is not None else []
        words, _ = self.top_words(main_word, reference, k=k, pos_tag=pos_tag, aligned=aligned)

        plot_vocab.extend(words)
        plot_vocab = list(set(plot_vocab))

        if ignore_words is not None:
            for word in ignore_words:
                if word in plot_vocab:
                    plot_vocab.remove(word)

        embeddings = []
        not_in_vocab = []

        for word in plot_vocab:
            if word != main_word:
                try:
                    embeddings.append(self.get_vector(word, reference, aligned=aligned))
                except ValueError:
                    not_in_vocab.append(word)
        for word in not_in_vocab:
            plot_vocab.remove(word)
        if chunks_tocompare is None:
            chunks_tocompare = list(range(len(self.chunk_indices)))
        else:
            for chunk in chunks_tocompare:
                if chunk not in self.chunk_indices[self._date_column].tolist():
                    raise ValueError(f"Chunk: {chunk} is not in the training chunks. Choose from: {self.chunk_indices[self._date_column]}")

            chunks_tocompare = [np.where(self.chunk_indices[self._date_column] == x)[0][0] for x in chunks_tocompare]

        main_word_embeddings = []
        # for chuck in range(len(self.chunks)):
        for chuck_idx in chunks_tocompare:
            try:
                vector = self.aligned_models[chuck_idx].wv.get_vector(main_word)
                main_word_embeddings.append(vector)
                plot_vocab.append(f'{main_word}_{self.chunk_indices[self._date_column].iloc[chuck_idx]}')
            except ValueError:
                raise ValueError(f"Word: {main_word} is not in the vocabulary of the model in chunk: {self.chunk_indices[self._date_column].iloc[chuck_idx]}")
        X = np.array(embeddings + main_word_embeddings)
        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("Number of samples must be greater than 1.")
        if tsne_perplexity > n_samples - 1:
            raise ValueError(f"Perplexity must be less than the number of samples - 1= {n_samples - 1}")
        X_embedded = TSNE(
            metric= tsne_metric,
            learning_rate= tsne_learning_rate,
            perplexity= tsne_perplexity
            ).fit_transform(X)

        df = pd.DataFrame(X_embedded)
        df['word'] = plot_vocab
        df_words = df.iloc[:-len(chunks_tocompare)]
        df_main_word = df.iloc[-len(chunks_tocompare):]

        # Plotting
        fig, ax = plt.subplots(1, figsize=(14,10), dpi=80, facecolor='#ffffff')
        ax.set_facecolor('#ffffff')

        # Plot the words' embeddings
        x_words = df_words[0]
        y_words = df_words[1]
        plt.scatter(x_words, y_words, c="0.6")

        for i in range(len(df_words)):
            ax.annotate(df_words['word'].iloc[i], (x_words.iloc[i], y_words.iloc[i]), color='#71797E', fontsize=14)

        # Plot the main_word's embeddings across all chunks
        x_main_word = df_main_word[0]
        y_main_word = df_main_word[1]
        plt.scatter(x_main_word, y_main_word, c="red")

        for i in range(len(df_main_word)):
            ax.annotate(df_main_word['word'].iloc[i], (x_main_word.iloc[i], y_main_word.iloc[i]), color='black', fontsize=17)

        # Draw arrows between main_word embeddings across chunks

        def drawArrow(A, B):
            plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], head_width=0.01, length_includes_head=True, color='black')
        
        for i in range(1, len(x_main_word)):
            drawArrow([x_main_word.iloc[i-1], y_main_word.iloc[i-1]], [x_main_word.iloc[i], y_main_word.iloc[i]])

        # Show the plot
        plt.savefig(f'{main_word}.png')
        plt.show()


    def save(self, path: str) -> None:
        """Saves the Word2Vec Semantic Shift model to the given path as a
        .pickle-file.

        Args:
            path: The path to which the Word2Vec Semantic Shift model should be saved.
        Returns:
            None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        pickle.dump(self, open(path, "wb"))

    def load(self, path: str) -> None:
        """Loads a pickled Word2Vec Semantic Shift model from the given path.

        Args:
            path: The path from which the Word2Vec Semantic Shift model should be loaded.
        Returns:
            None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        loaded = pickle.load(open(path, "rb"))
        self.__dict__ = loaded.__dict__

    def _get_top_k_words(
            self,
            model: Word2Vec,
            main_word: str,
            k: int = 10,
            pos_tagging: Union[bool, str, List[str]] = False
            ):
        """Get the top k most similar words to a word in the vocabulary of the
        model.

        Args:
            model: The Word2Vec model to get the top k most similar words of
            main_word: The word to get the top k most similar words of
            k: The number of words to return
            pos_tagging: The part-of-speech tag of the words to return
                         Can be a boolean, to return words with the same pos-tag
                         as the main word, a string, to return words with the
                         same pos-tag as the string, or a list of strings, to
                         return words with the same pos-tag as any of the strings

        Returns:
            topk: Tuple of lists of the top k most similar words and their cosine similarities
            similarity: The cosine similarities of the top k most similar words
        """
        try:
            if isinstance(pos_tagging, str):
                pos_tagging = [pos_tagging]
            elif isinstance(pos_tagging, bool) and pos_tagging:
                pos_tagging = [pos_tag([main_word])[0][1]]

            for multiplier in range(2, 10):
                sims = model.wv.most_similar(main_word, topn=k*multiplier)
                words, similarities = tuple(map(list, zip(*sims)))
                if pos_tagging:
                    words = [word for word in words if
                             pos_tag([word])[0][1] in pos_tagging]
                if len(words) >= k:
                    return words[:k], similarities[:k]
            raise ValueError(f"Could not find {k} words with the specified pos-tagging.")
        except KeyError:
            raise ValueError(
                f"Word '{main_word}' not in vocabulary. Please try another word.")

    def _procrustes_align(self, reference_model, aligned_model):
        """Original code: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
        Procrustes align two gensim Word2Vec models (to allow for comparison between same word across models).

        Args:
            reference_model: The reference Word2Vec model.
            aligned_model: The other Word2Vec model to be aligned.
        Returns:
            other_embed: The aligned Word2Vec model.
        """
        in_base_embed, in_other_embed = self._intersection_align(reference_model,
                                                                 aligned_model)
        in_base_embed.wv.fill_norms(force=True)
        in_other_embed.wv.fill_norms(force=True)
        base_vecs = in_base_embed.wv.get_normed_vectors()
        other_vecs = in_other_embed.wv.get_normed_vectors()

        svd_matrix = other_vecs.T.dot(base_vecs)
        u, _, v = np.linalg.svd(svd_matrix)
        orthogonal_matrix = u.dot(v)
        aligned_model.wv.vectors = aligned_model.wv.vectors.dot(
                                                    orthogonal_matrix)

        return aligned_model

    def _intersection_align(self, model1, model2):
        """Original code: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
        Intersect two gensim Word2Vec models, model1 and model2. Only the
        shared vocabulary between them is kept. Indices are reorganized in
        descending frequency.

        Args:
            model1: The first Word2Vec model.
            model2: The second Word2Vec model.
        Returns:
            model1: The first Word2Vec model with the aligned vocabulary.
            model2: The second Word2Vec model with the aligned vocabulary.
        """
        vocab_m1 = set(model1.wv.index_to_key)
        vocab_m2 = set(model2.wv.index_to_key)
        common_vocab = vocab_m1 & vocab_m2

        if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
            return model1, model2

        common_vocab = list(common_vocab)
        common_vocab.sort(key=lambda w: model1.wv.get_vecattr(w, "count") +
                                        model2.wv.get_vecattr(w, "count"),
                          reverse=True)

        for model in [model1, model2]:
            indices = [model.wv.key_to_index[w] for w in common_vocab]
            old_arr = model.wv.vectors
            new_arr = np.array([old_arr[index] for index in indices])
            model.wv.vectors = new_arr

            new_key_to_index = {}
            new_index_to_key = []
            for new_index, key in enumerate(common_vocab):
                new_key_to_index[key] = new_index
                new_index_to_key.append(key)
            model.wv.key_to_index = new_key_to_index
            model.wv.index_to_key = new_index_to_key

        return model1, model2

    def _prepare_for_training(self, texts: pd.DataFrame, text_column: str,
                              date_column: str, date_format: str,
                              update: bool = False) -> pd.DataFrame:
        """Prepares the texts for training.

        Args:
            texts: The texts to prepare.
            text_column: The column containing the texts.
            date_column: The column containing the dates.
            date_format: The format of the dates.
            update: Whether to update the model or not.
        """
        if not isinstance(texts, pd.DataFrame):
            raise TypeError("texts must be a pandas DataFrame!")
        if not isinstance(text_column, str):
            raise TypeError("text_column must be a string!")
        if text_column not in texts.columns:
            raise ValueError(
                "texts must contain the column specified in text_column!")
        if not isinstance(date_column, str):
            raise TypeError("date_column must be a string!")
        if date_column not in texts.columns:
            raise ValueError(
                "texts must contain the column specified in date_column!")
        if not isinstance(update, bool):
            raise TypeError("update must be a boolean!")
        self._text_column = text_column
        self._date_column = date_column

        if not isinstance(texts[self._text_column].iloc[0], list):
            raise TypeError(
                "The elements of the 'texts' column of texts must each contain a tokenized document as a list of strings!")

        texts[self._date_column] = pd.to_datetime(texts[self._date_column],
                                                  format=date_format)
        texts.sort_values(by=self._date_column, inplace=True)
        if update:
            new_chunks = _get_time_indices(texts, how=self.how,
                                                   date_column=self._date_column,
                                                   min_docs_per_chunk=self.min_docs_per_chunk)
            new_chunks["chunk_start"] += self._last_text
            self.chunk_indices = pd.concat([self.chunk_indices, new_chunks], ignore_index=True)
        else:
            self.chunk_indices = _get_time_indices(texts, how=self.how,
                                                   date_column=self._date_column,
                                                   min_docs_per_chunk=self.min_docs_per_chunk)
        self.sorting = texts.index
        return texts

    def _train_word2vec(self, texts: List[str], epochs: int = 5,
                        start_alpha: float = 0.025, end_alpha: float = 0.0001,
                        update: bool = False, **kwargs) -> Word2Vec:
        """Trainer function for the Word2Vec model.

        Args:
            texts: The texts to train the Word2Vec model on.
            epochs: The number of epochs to train the model.
            start_alpha: The initial learning rate.
            end_alpha: The final learning rate.
            update: Whether to update the model or not.
        Returns:
            The trained Word2Vec model.
        """
        model = Word2Vec(**self.trainer_args)
        phrases = Phrases(texts, min_count=self.bigram_min_count,
                          threshold=self.bigram_threshold)
        bigram = Phraser(phrases)
        sentences = bigram[texts]

        model.build_vocab(sentences, update=update)
        if len(model.wv.index_to_key) == 0:
            raise RuntimeError(
                "Vocabulary is empty, cannot proceed with training.")

        total_examples = model.corpus_count
        model.train(
            sentences,
            total_examples=total_examples,
            epochs=epochs,
            start_alpha=start_alpha,
            end_alpha=end_alpha,
            **kwargs
            )
        return model


    def _align_models(self, models: List[Word2Vec], reference: int = -1) -> None:
        """Aligns the Word2Vec models.

        Args:
            models: The Word2Vec models to align.
            reference: The reference chunk index.
        """
        self.aligned_models = []
        for i, model in enumerate(models):
            new_model = copy.deepcopy(model)
            reference_model = copy.deepcopy(models[reference])
            if i == reference or (i == len(models) + reference and reference < 0):
                self.aligned_models.append(new_model)
            self.aligned_models.append(self._procrustes_align(
                reference_model,
                new_model))
    def _check_inference_requirements(self, aligned):
        """Checks the requirements for inference.

        Args:
            aligned: Whether to check the requirements for aligned models or not.
        """
        if aligned:
            if not self.aligned_models:
                raise ValueError("Models are not aligned. Call fit or "
                                 "fit_update first.")
        else:
            if not self.word2vecs:
                raise ValueError("Models are not trained. Call fit or "
                                 "fit_update first.")
