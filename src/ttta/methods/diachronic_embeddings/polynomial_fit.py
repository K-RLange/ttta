
import random
from src.ttta.diachronic_embeddings.utils.components import WordSimilarities
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep
from typing import Literal
import warnings
from src.ttta.diachronic_embeddings.utils.settings import EmbeddingFiles, FileLoader
import json


class PolynomialFitting:
    def __init__(
            self,
            word:str
        ):
        self.files = EmbeddingFiles()
        self.word_proportions = []
        self.word = word
        self.years = self.files.years_used

        self.word_props, self.words = FileLoader.load_files(self.__class__.__name__, word)

        with open(f'../embeddings_similarity/embeddings_sim_w_.json') as f:
            self.word_props = json.load(f)

        for w_ in self.word_props:
             self.word_proportions += [WordSimilarities(**w_)]

        self.num_senses = len(self.word_proportions[0].props)


    def sense_distribution(self, sense_idx):
        if not sense_idx in range(0, self.num_senses + 1):
            raise ValueError(
                f'The sense index {sense_idx} not present in the range of senses available for the word {self.word}'
            )
        return [word_prop.props[sense_idx] for word_prop in self.word_proportions]

    def polynomial_fit(self, sense: int, deg:int=20):
        warnings.filterwarnings('ignore')
        dist = self.sense_distribution(sense)
        return np.poly1d(np.polyfit(self.years, dist, deg))

    def bspline_fit(self, sense:int):
        dist = self.sense_distribution(sense)
        tck_spline_args = splrep(self.years, dist, s=0, k=3)
        return BSpline(*tck_spline_args)

    def distribution_all_senses(self, fit:Literal['polynomial', 'bspline']):
        all_senses = []
        sense_ = {}
        xp = np.linspace(1980, 2018, 100)
        for sense_num in range(0, self.num_senses):
            sense_['years'] = self.years
            sense_['distribution'] = self.sense_distribution(sense_num)
            sense_['sense_id'] = sense_num

            if fit == 'polynomial':
                sense_['y_fit'] = self.polynomial_fit(sense_num)

            if fit == 'bspline':
                sense_['y_fit'] = self.bspline_fit(sense_num)(xp)

            all_senses += [sense_.copy()]

        return all_senses

def plot_word(word:str, fit:Literal['polynomial', 'bspline']):
    if not fit in ['polynomial', 'bspline']:
        raise ValueError(
            f'The fit type provided is not correct, expected "polynomial" or "bspline", got {type(fit)}'
        )

    fig, ax = plt.subplots()
    markers = ['o', 'v', '^', 's', 'p', 'P', 'h', 'H', 'D']
    random.shuffle(markers)
    poly_w1 = PolynomialFitting(word=word)
    xp = np.linspace(1980, 2018, 100)

    if fit == 'polynomial':
        distr_all_senses = poly_w1.distribution_all_senses(fit='polynomial')
        for sense, obj in zip(distr_all_senses, markers[:poly_w1.num_senses]):
            ax.plot(sense['years'], sense['distribution'], f'{obj}', label=f'{word}, for the sense: {sense["sense_id"]}')
            ax.plot(xp, sense['y_fit'](xp), '-')

    if fit == 'bspline':
        distr_all_senses = poly_w1.distribution_all_senses(fit='bspline')
        for sense, obj in zip(distr_all_senses, markers[:poly_w1.num_senses]):
            ax.plot(sense['years'], sense['distribution'], f'{obj}', label=f'{word}, for the sense: {sense["sense_id"]}')
            ax.plot(xp, sense['y_fit'], '-')

    ax.title.set_text(f'Word: {word}')
    ax.legend()
    plt.show()

def plot_words(words:tuple, sense_id_w1:int, sense_id_w2:int, sense_id_w3:int, fit:Literal['polynomial', 'bspline']):
    w_1, w_2, w_3= words

    poly_w1 = PolynomialFitting(word=w_1)
    poly_w2 = PolynomialFitting(word=w_2)
    poly_w3 = PolynomialFitting(word=w_3)

    xp = np.linspace(1980, 2018, 100)
    fig, ax = plt.subplots()
    dist_1 = poly_w1.sense_distribution(sense_idx=sense_id_w1)
    dist_2 = poly_w2.sense_distribution(sense_idx=sense_id_w2)
    dist_3 = poly_w3.sense_distribution(sense_idx=sense_id_w3)

    if fit == 'polynomial':
        p_1 = poly_w1.polynomial_fit(sense=sense_id_w1, deg=20)
        p_2 = poly_w2.polynomial_fit(sense=sense_id_w2, deg=20)
        p_3 = poly_w3.polynomial_fit(sense=sense_id_w3, deg=20)

        ax.plot(poly_w1.years, dist_1, '*', label=f'{w_1}, for the sense: {sense_id_w1}')
        ax.plot(xp, p_1(xp), '-', )
        ax.plot(poly_w1.years, dist_2, '*', label=f'{w_2} for the sense: {sense_id_w2}')
        ax.plot(xp, p_2(xp), '-',)

        ax.plot(poly_w1.years, dist_3, '+', label=f'{w_3} for the sense: {sense_id_w3}')
        ax.plot(xp, p_3(xp), '-', )

    if fit == 'bspline':
        b_1 = poly_w1.bspline_fit(sense=sense_id_w1)
        b_2 = poly_w2.bspline_fit(sense=sense_id_w2)
        b_3 = poly_w3.bspline_fit(sense=sense_id_w3)

        ax.plot(poly_w1.years, dist_1, '*', label=f'{w_1}, for the sense: {sense_id_w1}')
        ax.plot(xp, b_1(xp), '-', )
        ax.plot(poly_w1.years, dist_2, '*', label=f'{w_2} for the sense: {sense_id_w2}')
        ax.plot(xp, b_2(xp), '-', )

        ax.plot(poly_w1.years, dist_3, '+', label=f'{w_3} for the sense: {sense_id_w3}')
        ax.plot(xp, b_3(xp), '-', )


    ax.legend()
    plt.show()


if __name__ == '__main__':
    # plot_words(('abuse', 'black', 'kill'), sense_id_w1=2, sense_id_w2=2, sense_id_w3=2, fit='bspline')
    # plot_word('abuse', fit='bspline')
    p = PolynomialFitting(word='abuse')
    print(p.distribution_all_senses(fit='bspline'))