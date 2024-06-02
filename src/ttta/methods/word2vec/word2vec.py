from typing import Any, List, Union
from pathlib import Path
from gensim.models import Word2Vec
from pathlib import Path
from typing import List, Union
from nltk.tag import pos_tag
import numpy as np
# import nltk
# nltk.download('averaged_perceptron_tagger')


class Word2VecTrainer:  
    """
    Wrapper class for gensim.models.Word2Vec to train a Word2Vec model.

    Methods
    -------
        __init__(model_path, min_count, window, negative, ns_exponent, vector_size, workers, sg, **kwargs)
            Initialize the Word2Vec model
        fit(data, output_path, epochs, start_alpha, end_alpha, compute_loss, **kwargs)
            Train the Word2Vec model on the given data
    """  

    def __init__(
            self,
            **kwargs
            ):
        """
        Args:
            min_count (int, optional): Ignores all words with total frequency lower than this, by default 0
            window (int, optional): The maximum distance between the current and predicted word within a sentence, by default 15
            negative (int, optional): If > 0, negative sampling will be used, by default 5
            ns_exponent (float, optional): The exponent used to shape the negative sampling distribution, by default 0.75
            vector_size (int, optional): Dimensionality of the word vectors, by default 100
            workers (int, optional): Number of worker threads to train the model, by default 1
            sg (int, optional): Training algorithm: 1 for skip-gram; otherwise CBOW, by default 1
            **kwargs (optional): Additional arguments to pass to the gensim.models.Word2Vec constructor

        Attributes:
            model (gensim.models.Word2Vec): The Word2Vec model
        """
        try:
            self.model = Word2Vec(**kwargs)
        except Exception as e:
            raise f"Error initializing the Word2Vec model: {e}. Please check the input parameters. Refer to the gensim.models.Word2Vec documentation for more information."
        

    def copy(self):
        """
        Create a copy of the Word2VecTrainer object

        Returns:
            copy (Word2VecTrainer): A copy of the Word2VecTrainer object
        """
        

        copy = Word2VecTrainer()
        copy(model = self.model)
        return copy
    
    def __setattr__(self, name: str, value: Any) -> None:
        
        if name == 'model':
            if not isinstance(value, Word2Vec):
                raise ValueError("The model attribute must be of type gensim.models.Word2Vec")
        super().__setattr__(name, value)
    
    def train(
            self, 
            data: List[str],
            epochs=5,
            start_alpha=0.025,
            end_alpha=0.0001,
            compute_loss=True,
            update: bool = False, # update=True to update the model
            **kwargs
            ):
        """
        Train the Word2Vec model on the given data
        
        Args:
            data (List[str]): List of documents
            output_dir (Union[str, Path], None): Path to save the trained model, by default None
            epochs (int, optional): Number of epochs, by default 5
            start_alpha (float, optional): Learning rate, by default 0.025
            end_alpha (float, optional): Minimum learning rate, by default 0.0001
            compute_loss (bool, optional): Whether to compute the loss, by default True
            **kwargs : optional

        Examples:
            >>> from semantics.feature_extraction.word2vec import Word2VecTrainer
            >>> texts = ['This is a test.', 'This is another test.', 'This is a third test.']
            >>> Word2VecTrainer().train(texts, epochs=1)
            >>> print('Trained model: ', Word2VecTrainer().model)
            Trained model:  Word2Vec(vocab=5, vector_size=100, alpha=0.025)
        """
        from gensim.models.phrases import Phrases, Phraser
        sent = [doc.split() for doc in data]
        phrases = Phrases(sent, min_count=30)
        bigram = Phraser(phrases)
        sentences = bigram[sent]
        self.model.build_vocab(sentences, update=update)
        total_examples = self.model.corpus_count
        self.model.train(
                sentences,
                total_examples=total_examples,
                epochs=epochs,
                start_alpha=start_alpha,
                end_alpha=end_alpha,
                compute_loss=compute_loss,
                **kwargs
                )
        self.model.init_sims(replace=True)
        return self.model
    


    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save the trained model to a file

        Args:
            output_path (Union[str, Path]): Path to save the trained model
        """
        self.model.save(output_path)
    

    def load(self, model_path: Union[str, Path]) -> None:
        """
        Load a pretrained model from a file

        Args:
            model_path (Union[str, Path]): Path to the pretrained model
        """
        self.model = Word2Vec.load(model_path)

    
    def get_model(self) -> Word2Vec:
        """
        Get the trained model

        Returns:
            model (gensim.models.Word2Vec): The trained Word2Vec model
        """
        return self.model
    
    def get_vocab(self) -> List[str]:
        """
        Get the vocabulary of the trained model

        Returns:
            vocab (List[str]): The vocabulary of the model
        """
        return self.model.wv.index_to_key
    
    

class Word2VecAlign:
    """
    Wrapper class for gensim.models.Word2Vec to align Word2Vec models.

    Methods
    -------
        __init__(model_paths)
            Initialize the Word2VecAlign object with a list of paths to the Word2Vec models.
        load_models()
            Load the models
        align_models(reference_index, output_dir, method)
            Align the models
    """
    def __init__(
            self, 
            models: List[Word2Vec]
            ):
        """
        Args:
            model_paths (List[Word2Vec]): List of Word2Vec models to align. See gensim.models.Word2Vec for more information.
  
        """
        self.models = models

    def align(
            self,
            reference: int = -1,
            method: str = "procrustes",
            ) -> List[Word2Vec]:
        """
        Align the models

        Args: 
            reference (int, optional): Index of the reference model, by default -1
            output_dir (str, optional): Path to save the aligned models, by default None
            method (str, optional): Alignment method, by default "procrustes"
      
        Returns:
            aligned_models (List[gensim.models.Word2Vec]): List of aligned models

        Examples:
            >>> from semantics.feature_extraction.word2vec import Word2VecAlign
            >>> model_paths = ['model1.model', 'model2.model']
            >>> Word2VecAlign(model_paths).align_models(reference_index=0, output_dir='aligned_models')
            >>> print('Aligned models: ', Word2VecAlign(model_paths).aligned_models)
            Aligned models:  [Word2Vec(vocab=5, vector_size=100, alpha=0.025), Word2Vec(vocab=5, vector_size=100, alpha=0.025)]
        """
        
        if method != "procrustes":
            raise NotImplementedError("Only procrustes alignment is implemented. Please use method='procrustes'")

        
        self.reference_model = self.models[reference]
        self.aligned_models: List[Word2Vec] = []

        for i, model in enumerate(self.models):
            if i == reference:
                self.aligned_models.append(model)

            aligned_model = smart_procrustes_align_gensim(self.reference_model,model)
            self.aligned_models.append(aligned_model)
        return self.aligned_models


class Word2VecInference:
    """
    Wrapper class for gensim.models.Word2Vec for Inference.

    Methods
    -------
        __init__(pretrained_model_path)
            Initialize the Word2VecInference object with a pretrained model.
        get_embedding(word, norm)
            Infer the vector of a word
        get_similarity(word1, word2)
            Get the cosine similarity between two words
        get_top_k_words(word, k)
            Get the top k most similar words to a word in the vocabulary of the model.
    """
    def __init__(
            self,
            model: Word2Vec
            ):
        """
        Args:
            pretrained_model_path (str, optional): Path to a pretrained model, by default None  

        Attributes:
            word_vectorizor (Word2VecEmbeddings): The Word2VecEmbeddings object
        """
        self.model = model



    def infer_vector(self, word:str, norm = False) -> List[float]:
        """
        Infer the vector of a word

        Args:
            word (str): The word to infer the embedding vector of
            norm (bool, optional): Whether to normalize the vector, by default False

        Returns:
            embedding (List[float]): The embedding vector of the word
        """
        return self.model.wv.get_vector(word, norm = norm)
    
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Get the cosine similarity between two words' embedding vectors

        Args:
            word1 (str): The first word
            word2 (str): The second word
        
        Returns:
            similarity (float): The cosine similarity between the two words
        
        Examples:
            >>> from semantics.feature_extraction.word2vec import Word2VecInference
            >>> Word2VecInference('model.model').get_similarity('test', 'another')
            0.99999994
        """
        return self.model.wv.similarity(word1, word2)
    
    def get_top_k_words(
            self,
            main_word: str,
            k: int = 10,
            pot_tag: Union[bool, str, List[str]] = False
            ):
        """
        Get the top k most similar words to a word in the vocabulary of the model. Default k = 10

        Args:
            main_word (str): The word to get the top k most similar words of
            k (int, optional): The number of words to return, by default 10
        
        Returns:
            topk (Tuple[List[str], List[float]]): Tuple of lists of the top k most similar words and their cosine similarities
            similarity (List[float]): The cosine similarities of the top k most similar words
        
        Examples:
            >>> from semantics.feature_extraction.word2vec import Word2VecInference
            >>> Word2VecInference('model.model').get_top_k_words('test', k=1)
            (['another'], [0.9999999403953552])
        """

        
        try:
            top_k_words = []
            i = k
            if isinstance(pot_tag, str):
                pot_tag = [pot_tag]

            while len(top_k_words) < k:
                sims = self.model.wv.most_similar(main_word, topn= i)
                words, similarities= tuple(map(list, zip(*sims)))

               
                if isinstance(pot_tag, list):
                    words = [word for word in words if pos_tag([word])[0][1] in pot_tag]
                    
                elif pot_tag:
                    main_pos = pos_tag([main_word])[0][1]
                    words = [word for word in words if pos_tag([word])[0][1] == main_pos]

                top_k_words.extend(words)
                i += 1

            return top_k_words, similarities
        
        except KeyError:
            raise ValueError(f"Word '{main_word}' not in vocabulary. Please try another word.")
        








# ------------------- smart_procrustes_align_gensim --------------------
# from: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.

    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # re-filling the normed vectors
    in_base_embed.wv.fill_norms(force=True)
    in_other_embed.wv.fill_norms(force=True)

    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.wv.get_normed_vectors()
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    
    return other_embed

# ------------------- intersection_align_gensim --------------------
# from: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1,m2)
