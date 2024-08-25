from typing import List, Union, Optional, Tuple, Literal
import pandas as pd
from gensim.models import Word2Vec
from src.ttta.methods.word2vec.word2vec import Word2VecTrainer, Word2VecAlign, Word2VecInference
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Word2VecSemanticShift:
    def __init__(
            self,
            min_count: int = 2,
            window: int = 5,
            negative: int = 5,
            ns_exponent: float = 0.75,
            vector_size: int = 300,
            alpha=0.025,
            max_vocab_size=None, 
            sample=1e-3, 
            seed=1, 
            workers=3, 
            min_alpha=0.0001,
            sg=0, 
            hs=0,  
            cbow_mean=1, 
            null_word=0,
            trim_rule=None, 
            sorted_vocab=1, 
            compute_loss=False, 
            callbacks=(),
            comment=None, 
            max_final_vocab=None, 
            shrink_windows=True,
            ):
        

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

        self.trainers: List[Word2VecTrainer] = []
        self.word2vecs: List[Word2Vec] = []
        self.chunks: List[str] = []
        self.aligned_models: List[Word2Vec] = []

    def is_trained(self) -> bool:
        return len(self.word2vecs) > 0
    
    def get_chunks(self) -> List[str]:
        return self.chunks
    
    def get_reference(self) -> int:
        return self.reference

    def fit(
            self,
            texts: pd.DataFrame,
            text_column: str = "text",
            date_column: str = "date",
            date_format: str = "%Y-%m-%d",
            align_reference: int = -1,
            epochs=5,
            start_alpha=0.025,
            end_alpha=0.0001, 
            date_groupby: Optional[Literal["day", "week", "month", "year"]] = None
        ) -> None:

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
        
        if self.is_trained():
            raise ValueError("The model has already been trained. Call 'fit_update' to update the model.")

        texts[date_column] = pd.to_datetime(texts[date_column], format=date_format)
        if date_groupby:
            if date_groupby == "day":
                texts[date_column] = texts[date_column].dt.date
            elif date_groupby == "week":
                texts[date_column] = texts[date_column].dt.to_period('W').dt.to_timestamp()
            elif date_groupby == "month":
                texts[date_column] = texts[date_column].dt.to_period('M').dt.to_timestamp()
            elif date_groupby == "year":
                texts[date_column] = texts[date_column].dt.to_period('Y').dt.to_timestamp()
            else:
                raise ValueError("date_groupby must be one of 'day', 'week', 'month', or 'year'.")



        texts.sort_values(by=date_column, inplace=True)

        grouped_texts = texts.groupby(date_column)

        for date, group in grouped_texts:
            # print(f"Training on chunk: {date}")
            trainer = Word2VecTrainer(**self.trainer_args)
            trainer.train(group[text_column].tolist(), epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha)

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
            epochs=5,
            start_alpha=0.025,
            end_alpha=0.0001, 
        ):

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



    def _align_models(self, models: List[Word2Vec], reference: int = -1):
        self.aligner = Word2VecAlign(models)
        self.aligned_models = self.aligner.align(reference=reference)
        

    def save(self, file_dir: str, aligned: bool = True):
        """
        Saves the Word2Vec models to a file.
        Args:
            file_dir: The directory to save the Word2Vec models.
        """
        import os
        os.makedirs(file_dir, exist_ok=True)

        for idx in range(len(self.word2vecs)):
            if aligned:
                self.aligned_models[idx].save(f"{file_dir}/word2vec_{'_'.join(self.chunks[idx].split())}_aligned.model")

            else:
                self.word2vecs[idx].save(f"{file_dir}/word2vec_{'_'.join(self.chunks[idx].split())}.model")



    def load(self, file_dir: str, aligned: bool = True, align: bool = False, chunk_names: Optional[List[str]] = None, reference: Optional[int] = -1):
        """
        Loads the Word2Vec models from a file.
        Args:
            file_dir[str]: The directory to load the Word2Vec models.
            aligned[bool]: Whether the models are aligned or not. Default is True.
            align[bool]: Whether to align the models or not. Default is False.
            chunk_names[List[str]]: The names of the chunks. Default is None. If None, the names will be the index of the chunk.
            reference[int]: The reference chunk index. Default is None. If None, the reference will be the last chunk.

        """

        self.reference = reference

        import os
        if not os.path.exists(file_dir):
            raise FileNotFoundError(f"The directory: {file_dir} does not exist.")

        from glob import glob
        files = glob(f"{file_dir}/*.model")
        files = sorted(files)

        if not files or len(files) == 0:
            raise FileNotFoundError(f"No models found in the directory: {file_dir}")

        for i, file in enumerate(files):
            # print(f"Loading model: {file}, Chunk: {chunk_names[i] if chunk_names else i + 1}", end="\n\n")

            chunk = chunk_names[i] if chunk_names else i
            trainer = Word2VecTrainer.load(file)
            model = trainer.get_model()

            

            if aligned:
                self.aligned_models.append(model)
            else:
                self.word2vecs.append(model)

            if chunk not in self.chunks:
                self.chunks.append(chunk)

        if align:
            if not self.word2vecs:
                raise RuntimeError("No word2vec models found to align.")
            
            self._align_models(self.word2vecs, reference=reference)


    def get_parameters(self) -> dict:
        """
            Returns the parameters of the RollingLDA model.
            Returns:
                A dictionary containing the parameters of the RollingLDA model.
        """
        return self.__dict__.copy()


    def infer_vector(self, word: str, chunk_index, norm: bool = False, aligned: bool = True) -> np.ndarray:
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



    def top_words(self, word: str, chuck_index, k: int = 10, pos_tag: Union[bool, str, List[str]] = False, aligned: bool = True) -> Tuple[List[str], List[float]]:
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
                    # print(f"Word: {word} is not in the vocabulary of the model.")
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
