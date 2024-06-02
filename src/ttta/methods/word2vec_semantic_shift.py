from typing import List, Union, Optional
import pandas as pd
import pickle
from gensim.models import Word2Vec
from src.ttta.methods.word2vec.word2vec import Word2VecTrainer, Word2VecAlign, Word2VecInference
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Word2VecSemanticShift:
    def __init__(
            self,
            min_count: int = 20,
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
        

        self.trainer = Word2VecTrainer(
            min_count=min_count,
            window=window,
            negative=negative,
            ns_exponent=ns_exponent,
            vector_size=vector_size,
            alpha=alpha,
            max_vocab_size=max_vocab_size,
            sample=sample,
            seed=seed,
            workers=workers,
            min_alpha=min_alpha,
            sg=sg,
            hs=hs,
            cbow_mean=cbow_mean,
            null_word=null_word,
            trim_rule=trim_rule,
            sorted_vocab=sorted_vocab,
            compute_loss=compute_loss,
            callbacks=callbacks,
            comment=comment,
            max_final_vocab=max_final_vocab,
            shrink_windows=shrink_windows
        )

        self.trainers = None
        self.word2vecs = None
        self.aligned_word2vecs = None
        self.chunk_indices = None
        self.reference = None

    def is_trained(self) -> bool:
        return self.word2vecs is not None

    def fit(
            self,
            texts: pd.DataFrame,
            text_column: str = "text",
            date_column: str = "date",
            align_reference: int = -1,
            epochs=5,
            start_alpha=0.025,
            end_alpha=0.0001, 
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

        texts[date_column] = pd.to_datetime(texts[date_column])
        texts.sort_values(by=date_column, inplace=True)

        grouped_texts = texts.groupby(date_column)
        self.trainers: List[Word2VecTrainer] = []
        self.word2vecs: List[Word2Vec] = []
        self.chunk_indices: List[str] = []

        for date, group in grouped_texts:
            trainer = self.trainer.copy()
            trainer.train(group[text_column].tolist(), epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha)

            self.trainers.append(trainer)
            self.word2vecs.append(trainer.get_model())
            self.chunk_indices.append(str(date))

        
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

        if self.trainer is None:
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
                idx = self.chunk_indices.index(date)
            except ValueError:
                raise ValueError(f"The date: {date}, is not in the existing chunk indices. Call fit instead.")
            
            self.trainers[idx].train(group[text_column].tolist(), epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, update=True)
            self.word2vecs[idx] = self.trainers[idx].get_model()


        self.reference = align_reference
        self._align_models(self.word2vecs, reference=align_reference)



    def _align_models(self, models: List[Word2Vec], reference: int = -1):
        self.aligner = Word2VecAlign(models)
        self.aligned_models = self.aligner.align(reference=reference)
        

    def save(self, file_dir: str):
        state = {
            'trainers': self.trainers,
            'word2vecs': self.word2vecs,
            'aligned_word2vecs': self.aligned_word2vecs,
            'chunk_indices': self.chunk_indices,
            'reference': self.reference
        }

        with open(file_dir, 'wb') as f:
            pickle.dump(state, f)

    def load(self, file_dir: str):
        with open(file_dir, 'rb') as f:
            state = pickle.load(f)

        self.trainers = state['trainers']
        self.word2vecs = state['word2vecs']
        self.aligned_word2vecs = state['aligned_word2vecs']
        self.chunk_indices = state['chunk_indices']
        self.reference = state['reference']



    def get_parameters(self) -> dict:
        """
            Returns the parameters of the RollingLDA model.
            Returns:
                A dictionary containing the parameters of the RollingLDA model.
        """
        return self.__dict__.copy()


    def infer_vector(self, word: str, chunk_index, norm: bool = False) -> List[float]:
        if not self.aligned_models:
            raise ValueError("Models are not aligned. Call fit or fit_update first.")

        inferencer = Word2VecInference(self.aligned_models[chunk_index])
        return inferencer.infer_vector(word, norm=norm)



    def top_words(self, word: str, chuck_index, k: int = 10, pos_tag: Union[bool, str, List[str]] = False) -> List[str]:
        if not self.aligned_models:
            raise ValueError("Models are not aligned. Call fit or fit_update first.")

        inferencer = Word2VecInference(self.aligned_models[chuck_index])
        top_words, similarities = inferencer.get_top_k_words(word, k=k, pos_tag=pos_tag)
        return top_words, similarities



    def visualize(self, chuck_index, main_word: str, k: int = 10, pos_tag: Union[bool, str, List[str]] = False, extra_words: Optional[List[str]] = None):
        if not self.aligned_models:
            raise ValueError("Models are not aligned. Call fit or fit_update first.")

        
        plot_vocab = extra_words if extra_words is not None else []
        last_chuck = chuck_index[-1]

        inferencer = Word2VecInference(self.aligned_models[last_chuck])
        try:
            words, _ = inferencer.get_top_k_words(main_word, k=k, pos_tag=pos_tag)
        except ValueError:
            raise ValueError(f"Word: {main_word} is not in the vocabulary of the model.")
        
        
        plot_vocab = plot_vocab + words
        plot_vocab = list(set(plot_vocab))

        embeddings = []
        not_in_vocab = []

        for word in plot_vocab:
            if word != main_word:
                try:
                    vector = inferencer.infer_vector(word)
                    embeddings.append(vector)
                
                except ValueError:
                    print(f"Word: {word} is not in the vocabulary of the model.")
                    not_in_vocab.append(word)
        
        main_word_embeddings = []
        for chuck in chuck_index:
            inferencer = Word2VecInference(self.aligned_models[chuck])
            try:
                vector = inferencer.infer_vector(main_word)
                main_word_embeddings.append(vector)
            except ValueError:
                raise ValueError(f"Word: {main_word} is not in the vocabulary of the model in chunk: {self.chunk_indices[chuck]}")
            

        n_components = 2
        metric='euclidean'
        learning_rate='auto'

        X = np.array(embeddings)
        X_embedded = TSNE(n_components=n_components, metric=metric, learning_rate=learning_rate).fit_transform(X)


        
        df = pd.DataFrame(X_embedded)

        df2 = df.iloc[:-len(chuck_index)]
        df3 = df.iloc[-len(chuck_index):]

        df3 = df3.reset_index()

        plot_vocab1 = plot_vocab[:-len(chuck_index)]
        plot_vocab2 = plot_vocab[-len(chuck_index):]

        x = df2[0] 
        y = df2[1]
        x2 = df3[0]
        y2 = df3[1]

        # just some size adjustment...
        fig, ax = plt.subplots(1, figsize=(14,10),dpi=80,facecolor='#ffffff')
        ax.set_facecolor('#ffffff')

        # plot x and y
        plt.scatter(x,y,c="0.6")

        textsvocab = [plt.text(x[i], y[i], txt,color='#71797E',fontsize = 14) for i,txt in enumerate(plot_vocab1)]
        plt.scatter(x2,y2,c="red")
        textsvocab2 = [plt.text(x2[i], y2[i], txt,color='black',fontsize = 17) for i,txt in enumerate(plot_vocab2)]

        def drawArrow(A, B):
            plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], head_width=0.01, length_includes_head=True,color='black')

        for i in range(1,len(chuck_index)):
            X = np.array([x2[len(x2)-i],y2[len(y2)-i]])
            Y = np.array([x2[len(x2)-(i+1)],y2[len(y2)-(i+1)]])
            drawArrow(Y,X)

        #show the plot
        plt.savefig(f'{main_word}.png')
        plt.show()

        


