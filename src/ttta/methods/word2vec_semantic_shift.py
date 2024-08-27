import pickle
import warnings
from typing import List, Union, Optional, Tuple, Literal
import pandas as pd
from gensim.models import Word2Vec
from ttta.methods.word2vec.word2vec import Word2VecTrainer, Word2VecAlign, Word2VecInference
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from ttta.preprocessing.chunk_creation import _get_time_indices
from datetime import datetime

class Word2VecSemanticShift:
    """
    Word2VecSemanticShift class for training and aligning Word2Vec models for semantic shift analysis.

    Methods:
        fit: Fits the Word2Vec models on the texts.
        fit_update: Updates the Word2Vec models on the texts.
        _align_models: Aligns the Word2Vec models.
        save: Saves this class to a file
        load: Loads this class from a file.
        get_parameters: Returns the parameters of the RollingLDA model.
        infer_vector: Infers the vector of a word.
        top_words: Returns the top words similar to a word.
        visualize: Visualizes the semantic shift of a word across chunks
    """
    def __init__(
            self,
            how: Union[str, List[datetime]] = "M",
            min_count: int = 2,
            min_docs_per_chunk: int = None,
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
        self.reference = None
        self.how = how
        self.trainers: List[Word2VecTrainer] = []
        self.word2vecs: List[Word2Vec] = []
        self.chunks: List[str] = []
        self.aligned_models: List[Word2Vec] = []
        self._verbose = verbose
        self.chunk_indices = None
        self.min_docs_per_chunk = min_docs_per_chunk

    def is_trained(self) -> bool:
        """
        Returns whether the model has been trained or not.
        Returns:
            A boolean value indicating whether the model has been trained or not.
        
        Examples:
            >>> ss = Word2VecSemanticShift()
            >>> ss.is_trained()
            False
        """
        return len(self.word2vecs) > 0
    
    def get_chunks(self) -> List[str]:
        """
        Returns the chunks of the model.
        Returns:
            A list of the chunks of the model.
        """
        return self.chunks
    
    def get_reference(self) -> int:
        """
        Returns the reference chunk index.
        Returns:
            The reference chunk index.
        """
        return self.reference

    def fit(
            self,
            texts: pd.DataFrame,
            text_column: str = "text",
            date_column: str = "date",
            date_format: str = "%Y-%m-%d",
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


        if isinstance(texts[self._text_column].iloc[0], list):
            if not isinstance(texts[self._text_column].iloc[0][0], str):
                raise TypeError("text column must be contain list of strings or lists of lists of strings!")
        elif isinstance(texts[self._text_column].iloc[0], str):
            texts[self._text_column] = texts[self._text_column].apply(lambda x: x.split())
            warnings.warn("The elements of the text column contain a string. "
                          "Beware that the texts might not be preprocessed")
        else:
            raise TypeError(
                "text column must be contain list of strings or lists of lists of strings!")
        
        if self.is_trained():
            raise ValueError("The model has already been trained. Call 'fit_update' to update the model.")

        texts[self._date_column] = pd.to_datetime(texts[self._date_column], format=date_format)
        texts.sort_values(by=self._date_column, inplace=True)
        self.chunk_indices = _get_time_indices(texts, how=self.how, date_column=self._date_column, min_docs_per_chunk=self.min_docs_per_chunk)

        iterator = self.chunk_indices.iterrows()
        if self._verbose > 0:
            iterator = tqdm(iterator, total=len(self.chunk_indices), unit="chunk")
        for i, row in enumerate(iterator):
            date = self.chunk_indices["Date"].iloc[i]
            trainer = Word2VecTrainer(**self.trainer_args)
            if self._verbose > 0 and i < len(self.chunk_indices) - 1:
                iterator.set_description(f"Chunk {i + 1}/{len(self.chunk_indices)}")
            if i == 0:  # fit warmup chunks
                trainer.train(texts[self._text_column].iloc[:self.chunk_indices["chunk_start"].iloc[1]],
                              epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha)
                continue
            end = len(texts) if i + 1 >= len(self.chunk_indices) else int(self.chunk_indices.iloc[i + 1]["chunk_start"] - 1)
            trainer.train(texts[text_column].iloc[self.chunk_indices["chunk_start"].iloc[i]:end], epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha)

            self.trainers.append(trainer)
            self.word2vecs.append(trainer.get_model())
            self.chunks.append(str(date))

        
        self.reference = align_reference
        self._align_models(self.word2vecs, reference=align_reference)

    def fit_update(
            self,
            texts: pd.DataFrame,
            text_column: str = "text",
            date_column: str = "date",
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
        
        Examples:
            >>> ss = Word2VecSemanticShift()
            >>> data = pd.read_csv("data.csv")
            >>> ss.fit_update(data, text_column="text", date_column="date", align_reference=-1, epochs=5, start_alpha=0.025, end_alpha=0.0001)
        """

        if self.word2vecs is None:
            raise ValueError("Initial training has not been done. Call fit first.")
        
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

        if not isinstance(texts[self._text_column].iloc[0], str):
            raise TypeError("The elements of the 'texts' column of texts must each contain a string!")
        
        if not self.is_trained():
            raise ValueError("The model has not been trained. Call 'fit' to train the model first.")

        texts[date_column] = pd.to_datetime(texts[date_column])
        texts.sort_values(by=date_column, inplace=True)

        grouped_texts = texts.groupby(date_column)
        for date, group in grouped_texts:
            try:
                idx = self.chunks.index(str(date))
            except ValueError:
                raise ValueError(f"The date: {str(date)}, is not in the training chunks. Call fit instead.")
            # print(f"Training on chunk: {date}")
            self.trainers[idx].train(group[text_column].tolist(), epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, update=True)
            self.word2vecs[idx] = self.trainers[idx].get_model()


        self.reference = align_reference
        self._align_models(self.word2vecs, reference=align_reference)

    def _align_models(self, models: List[Word2Vec], reference: int = -1) -> None:
        """
        Aligns the Word2Vec models.
        Args:
            models: The Word2Vec models to align.
            reference: The reference chunk index.
        """
        self.aligner = Word2VecAlign(models)
        self.aligned_models = self.aligner.align(reference=reference)

    def save(self, path: str) -> None:
        """
            Saves the Word2Vec Semantic Shift model to the given path as a .pickle-file.
            Args:
                path: The path to which the Word2Vec Semantic Shift model should be saved.
            Returns:
                None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        pickle.dump(self, open(path, "wb"))

    def load(self, path: str) -> None:
        """
            Loads a pickled Word2Vec Semantic Shift model from the given path.
            Args:
                path: The path from which the Word2Vec Semantic Shift model should be loaded.
            Returns:
                None
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string!")
        loaded = pickle.load(open(path, "rb"))
        self.__dict__ = loaded.__dict__

    def get_parameters(self) -> dict:
        """
        Returns the parameters of the Word2Vec models.
        Returns:
            A dictionary containing the parameters of the Word2Vec models.
        
        Examples:
            >>> ss = Word2VecSemanticShift()
            >>> ss.get_parameters()
            {'min_count': 2, 'window': 5, 'negative': 5, 'ns_exponent': 0.75, 'vector_size': 300, 'alpha': 0.025, 'max_vocab_size': None, 'sample': 0.001, 'seed': 1, 'workers': 3, 'min_alpha': 0.0001, 'sg': 0, 'hs': 0, 'cbow_mean': 1, 'null_word': 0, 'trim_rule': None, 'sorted_vocab': 1, 'compute_loss': False, 'callbacks': None, 'comment': None, 'max_final_vocab': None, 'shrink_windows': False}
        
        """
        return self.__dict__.copy()

    def infer_vector(self, word: str, chunk_index: int, norm: bool = False, aligned: bool = True) -> np.ndarray:
        """
        Infers the vector of a word.
        Args:
            word: The word to infer the vector.
            chunk_index: The index of the chunk.
            norm: Whether to normalize the vector or not.
            aligned: Whether to infer the vector from the aligned models or not.
        Returns:
            The inferred vector of the word.
        
        Examples:
            >>> ss = Word2VecSemanticShift()
            >>> ss.infer_vector("trump", 0, aligned=True)
            array([ 0.001,  0.002,  0.003, ..., -0.001,  0.002,  0.001], dtype = float32)
        """
        if aligned:
            if not self.aligned_models:
                raise ValueError("Models are not aligned. Call fit or fit_update first.")

            inferencer = Word2VecInference(self.aligned_models[chunk_index])
            return inferencer.infer_vector(word, norm=norm)
        
        else:
            if not self.word2vecs:
                raise ValueError("Models are not trained. Call fit or fit_update first.")

            inferencer = Word2VecInference(self.word2vecs[chunk_index])
            return inferencer.infer_vector(word, norm=norm)

    def top_words(self, word: str, chuck_index: int, k: int = 10, pos_tag: Union[bool, str, List[str]] = False, aligned: bool = True) -> Tuple[List[str], List[float]]:
        """
        Returns the top words similar to a word.
        Args:
            word: The word to get the top words.
            chuck_index: The index of the chunk.
            k: The number of top words to return.
            pos_tag: The part-of-speech tag of the words to return.
            aligned: Whether to get the top words from the aligned models or not.
        Returns:
            A tuple containing the top words and their similarities.
        
        Examples:
            >>> ss = Word2VecSemanticShift()
            >>> ss.top_words("trump", 0, k=2, pos_tag=False, aligned=True)
            (['donald', 'president'], [0.8, 0.7])
        """
        if aligned:
            if not self.aligned_models:
                raise ValueError("Models are not aligned. Call fit or fit_update first.")

            inferencer = Word2VecInference(self.aligned_models[chuck_index])
            top_words, sims = inferencer.get_top_k_words(word, k=k, pos_tag=pos_tag)
            return top_words, sims
    
        else:
            if not self.word2vecs:
                raise ValueError("Models are not trained. Call fit or fit_update first.")

            inferencer = Word2VecInference(self.word2vecs[chuck_index])
            top_words, sims = inferencer.get_top_k_words(word, k=k, pos_tag=pos_tag)
            return top_words, sims
       
    def get_vocab(self, chunk: Union[int,str] = "reference") -> List[str]:
        """
        Returns the vocabulary of a chunk.
        Args:
            chunk: The index of the chunk or 'reference'.
        Returns:
            The vocabulary of the chunk.
        """
        if chunk == "reference":
            return self.word2vecs[-1].wv.index_to_key
        elif isinstance(chunk, int):
            return self.word2vecs[chunk].wv.index_to_key
        else:
            raise ValueError("chunk must be an integer or 'reference'.")

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
        """
        Visualizes the semantic shift of a word across chunks.
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
        
        Examples:
            >>> ss = Word2VecSemanticShift()
            >>> ss.visualize(
            ...     reference=-1,
            ...     chunks_tocompare=['1980', '2000', '2015', '2017'],
            ...     main_word="trump",
            ...     k=30,
            ...     pos_tag=False,
            ...     extra_words= [
            ...         "game",
            ...         "play",
            ...         "king",
            ...         "jack",
            ...         "heart"
            ...     ],
            ...     ignore_words= [
            ...         "use",
            ...         "give",
            ...         "would"
            ...     ],
            ...     aligned=True,
            ...     tsne_perplexity=10,
            ...     tsne_metric='euclidean',
            ...     tsne_learning_rate='auto'
            ... )
        """
        if aligned:
            if len(self.aligned_models) == 0:
                raise ValueError("Models are not aligned. Call fit or fit_update first.")

            inferencer = Word2VecInference(self.aligned_models[reference])
        else:
            if len(self.word2vecs) == 0:
                raise ValueError("Models are not trained. Call fit or fit_update first.")

            inferencer = Word2VecInference(self.word2vecs[reference])

        plot_vocab = extra_words if extra_words is not None else []

        try:
            words, _ = inferencer.get_top_k_words(main_word, k=k, pos_tag=pos_tag)
        
        except ValueError:
            raise ValueError(f"Word: {main_word} is not in the vocabulary of the model.")
        
        
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
                    vector = inferencer.infer_vector(word)
                    embeddings.append(vector)
                
                except ValueError:
                    not_in_vocab.append(word)

        for word in not_in_vocab:
            plot_vocab.remove(word)

        if chunks_tocompare is None:
            chunks_tocompare = list(range(len(self.chunks)))
        
        else:
            for chunk in chunks_tocompare:
                if chunk not in self.chunks:
                    raise ValueError(f"Chunk: {chunk} is not in the training chunks. Choose from: {self.chunks}")

            chunks_tocompare = [self.chunks.index(chunk) for chunk in chunks_tocompare]

        main_word_embeddings = []
        # for chuck in range(len(self.chunks)):
        for chuck_idx in chunks_tocompare:
            inferencer = Word2VecInference(self.aligned_models[chuck_idx])
            try:
                vector = inferencer.infer_vector(main_word)
                main_word_embeddings.append(vector)
                plot_vocab.append(f'{main_word}_{self.chunks[chuck_idx]}')

            except ValueError:
                raise ValueError(f"Word: {main_word} is not in the vocabulary of the model in chunk: {self.chunks[chuck_idx]}")
        
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
