
import os
import json
from numpy.linalg import norm
from ttta.preprocessing.schemas import Embedding, WordSenseEmbedding, SenseEmbedding
from ttta.preprocessing.settings import EmbeddingFiles, FileLoader
import logging
import numpy as np
from collections import Counter
from typing import List
from pydantic import parse_obj_as

class Similarities:
    """Class that computes the similarity between the embeddings of the words
    and the embeddings of the senses.

    Attributes
    ----------
        files: EmbeddingFiles
            EmbeddingFiles object that contains the paths to the files
        embedding_component: Embedding
            Embedding object that contains the word and the embeddings
        sense_component: SenseEmbedding
            SenseEmbedding object that contains the sense id and the embeddings
        embeddings_senses: List[Dict]
            List of dictionaries that contains the word and the senses
        words: str
            The list of all the words in the embeddings_senses file
        embeddings_examples: List[Dict]
            List of dictionaries that contains the word and the embeddings

    Methods
    -------
        _lookup_examples_that_match(words: str) -> List[Embedding]
            Returns the embeddings of the words that are present in the embeddings_examples file
        _search_word_sense(main_word: str) -> List[SenseEmbedding]
            Returns the senses of the word that are present in the embeddings_senses file
        _cos_sim(vect_a: np.array, vect_b: np.array) -> float
            Returns the cosine similarity between two vectors
        __call__(main_word: str, year: int, path_embeddings_file: str) -> None
            Computes the similarity between the embeddings of the words and the embeddings of the senses
    """
    def __init__(
            self,
        ):

        self.files = EmbeddingFiles()
        self.embedding_component = Embedding()
        self.embeddings_senses, self.words = FileLoader.load_files(self.__class__.__name__)

        self.embeddings_examples = None
        self.root_dir = self.files.embeddings_root_dit


    def _lookup_examples_that_match(self) -> List[Embedding]:
        """
        Returns the embeddings of the words that are present in the embeddings_examples file
        Returns:
            List[Embedding]

        """
        if self.embeddings_examples is None:
            raise ValueError(
                'Embedding examples not initialized'
            )

        embeddings = [embed.word for embed in self.embeddings_examples]
        for word in list(set(embeddings) & set(self.words.split('\n'))):
            yield next(e for e in self.embeddings_examples if word == e.word)


    def _search_word_sense(self, main_word:str) -> List[WordSenseEmbedding]:
        """
        Returns the senses of the word that are present in the embeddings_senses file
        Args:
            main_word: str

        Returns:
            List[SenseEmbedding]
        """
        for w in self.embeddings_senses:
            if not w.word == main_word:
                continue

            return w.senses

    def _cos_sim(self, vect_a:np.array, vect_b:np.array):
        """
        Returns the cosine similarity between two vectors
        Args:
            vect_a: np.array
            vect_b: np.array

        Returns:
            float
        """
        return (vect_a @ vect_b)/(norm(vect_a) * norm(vect_b))

    def __call__(self, main_word:str, year:int, path_embeddings_file:str):
        w_senses = self._search_word_sense(main_word)
        if len(w_senses) == 0:
            raise TypeError(f'The word {main_word} is not present in the embeddings_senses file')

        print(f'{"-" * 10} Computing Similarities for the word {main_word} {"-" * 10}')

        with open(os.path.join(self.root_dir, path_embeddings_file), 'r') as f:
            logging.info(f'{"-" * 10} Loading the embeddings examples file: {f.name} {"-" * 10}')
            self.embeddings_examples = parse_obj_as(List[Embedding], json.load(f))

        self.all_embeddings = list(self._lookup_examples_that_match())

        all_sims = []
        for embed in self.all_embeddings:
            for ex in embed.embeddings:
                s_argmax = np.argmax([self._cos_sim(np.array(ex), np.array(s.embedding)) for s in w_senses])
                all_sims += [w_senses[s_argmax].id]

        return list(map(lambda x: x[1]/len(all_sims), Counter(all_sims).most_common()))


def sim_on_all_words():
    """
    Computes the similarity between the embeddings of the words and the embeddings of the senses
    Returns:
        None

    """
    sim = Similarities()
    word_proportions = {}

    with open(sim.files.poly_words_f, 'r') as f:
        words = f.read()

    for w_ in words.split('\n'):
        s = []
        w_senses = sim._search_word_sense(w_)
        word_proportions['word'] = w_
        word_proportions['sense_ids'] = [w.id for w in w_senses]
        for year in sim.files.years_used:
            s += [{'year': year, 'props': sim(w_, year, path_embeddings_file=f'embeddings_{year}.json')}.copy()]

        word_proportions['props'] = s
        with open(f'../embeddings_similarity/embeddings_sim_{w_}.json', 'w') as f:
            json.dump(word_proportions, f, indent=4)


if __name__ == '__main__':
    sim_on_all_words()

