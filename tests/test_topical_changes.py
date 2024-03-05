import os
import shutil
import unittest
import numpy as np
import pandas as pd
import sys
sys.path.append('../src')
from ttta.methods.rolling_lda import RollingLDA
from ttta.methods.topical_changes import TopicalChanges

class TestTopicalChanges(unittest.TestCase):
    def setUp(self):
        self.texts = [["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"],
                      ["sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore", "magna"],
                      ["aliqua", "ut", "enim", "ad", "minim", "veniam", "quis", "nostrud", "exercitation", "ullamco"],
                      ["laboris", "nisi", "ut", "aliquip", "ex", "ea", "commodo", "consequat"],
                      ["duis", "aute", "irure", "dolor", "in", "reprehenderit", "in", "voluptate", "velit", "esse"],
                      ["cillum", "dolore", "eu", "fugiat", "nulla", "pariatur"],
                      ["excepteur", "sint", "occaecat", "cupidatat", "non", "proident", "sunt", "in", "culpa", "qui"],
                      ["officia", "deserunt", "mollit", "anim", "id", "est", "laborum"],
                      ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"],
                      ["sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore", "magna"],
                      ["aliqua", "ut", "enim", "ad", "minim", "veniam", "quis", "nostrud", "exercitation", "ullamco"],
                      ["laboris", "nisi", "ut", "aliquip", "ex", "ea", "commodo", "consequat"],
                      ["duis", "aute", "irure", "dolor", "in", "reprehenderit", "in", "voluptate", "velit", "esse"],
                      ["cillum", "dolore", "eu", "fugiat", "nulla", "pariatur"],
                      ["excepteur", "sint", "occaecat", "cupidatat", "non", "proident", "sunt", "in", "culpa", "qui"],
                      ["officia", "deserunt", "mollit", "anim", "id", "est", "laborum"],
                      ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"],
                      ["sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore", "magna"],
                      ["aliqua", "ut", "enim", "ad", "minim", "veniam", "quis", "nostrud", "exercitation", "ullamco"],
                      ["laboris", "nisi", "ut", "aliquip", "ex", "ea", "commodo", "consequat"],
                      ["duis", "aute", "irure", "dolor", "in", "reprehenderit", "in", "voluptate", "velit", "esse"]]
        self.dates = [pd.to_datetime("2021-01-01"), pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-03")] * 7
        self.df = pd.DataFrame({"text": self.texts, "date": self.dates})
        self.df.sort_values("date", inplace=True)
        self.roll = RollingLDA(2.0, prototype=3.0, how="1D", warmup=1.0, min_docs_per_chunk=1.0, verbose=0.0,
                          topic_threshold=[1.0, 0.0], initial_epochs=20.0, subsequent_epochs=10.0,
                          memory=1.0, min_count=1.0)
        self.roll.fit(self.df)

    def test_init(self):
        change_obj = TopicalChanges(self.roll, mixture="0.9", reference_period=4.0, quantile_threshold="0.8", samples=5.0)
        self.assertIsInstance(change_obj, TopicalChanges)
        self.assertEqual(change_obj._roll, self.roll)
        self.assertEqual(change_obj.mixture, 0.9)
        self.assertEqual(change_obj.reference_period, 4)
        self.assertEqual(change_obj.quantile_threshold, 0.8)
        self.assertEqual(change_obj.samples, 5)

    def test_init_invalid(self):
        with self.assertRaises(TypeError):
            TopicalChanges(1)
        with self.assertRaises(TypeError):
            TopicalChanges(self.roll, mixture="a")
        with self.assertRaises(ValueError):
            TopicalChanges(self.roll, mixture=1.1)
        with self.assertRaises(ValueError):
            TopicalChanges(self.roll, mixture=-1.1)
        with self.assertRaises(ValueError):
            TopicalChanges(self.roll, reference_period=0)
        with self.assertRaises(TypeError):
            TopicalChanges(self.roll, reference_period=1.5)
        with self.assertRaises(TypeError):
            TopicalChanges(self.roll, reference_period="a")
        with self.assertRaises(TypeError):
            TopicalChanges(self.roll, quantile_threshold="a")
        with self.assertRaises(ValueError):
            TopicalChanges(self.roll, quantile_threshold=-0.5)
        with self.assertRaises(ValueError):
            TopicalChanges(self.roll, quantile_threshold=1.5)
        with self.assertRaises(TypeError):
            TopicalChanges(self.roll, samples="a")
        with self.assertRaises(ValueError):
            TopicalChanges(self.roll, samples=0)
        with self.assertRaises(TypeError):
            TopicalChanges(self.roll, samples=1.5)
        with self.assertRaises(TypeError):
            TopicalChanges(self.roll, save_path=5)
        with self.assertRaises(TypeError):
            TopicalChanges(self.roll, load_path=5)
        with self.assertRaises(TypeError):
            TopicalChanges(self.roll, fast=5)

    def test_save_and_load(self):
        change_obj1 = TopicalChanges(self.roll, save_path="topical_changes_test", samples=5)
        change_obj2 = TopicalChanges(self.roll, load_path="topical_changes_test", samples=5)
        self.assertEqual(change_obj1._roll, change_obj2._roll)
        self.assertEqual(change_obj1.mixture, change_obj2.mixture)
        self.assertEqual(change_obj1.reference_period, change_obj2.reference_period)
        self.assertEqual(change_obj1.quantile_threshold, change_obj2.quantile_threshold)
        self.assertEqual(change_obj1.samples, change_obj2.samples)
        self.assertTrue(np.allclose(change_obj1._distances_observed, change_obj2._distances_observed))
        self.assertTrue(np.allclose(change_obj1._distances_simulated, change_obj2._distances_simulated))
        shutil.rmtree("topical_changes_test")

    def test_sample_invalid(self):
        change_obj = TopicalChanges(self.roll, samples=50)
        with self.assertRaises(TypeError):
            change_obj._sample(1, 1, np.array([1] * 2))
        with self.assertRaises(TypeError):
            change_obj._sample(np.array([0.5] * len(self.roll.lda.get_vocab())), 1.5, np.array([1] * 2))
        with self.assertRaises(ValueError):
            change_obj._sample(np.array([0.5] * len(self.roll.lda.get_vocab())), -1, np.array([1] * 2))
        with self.assertRaises(TypeError):
            change_obj._sample(np.array([0.5] * len(self.roll.lda.get_vocab())), 1, 1)

    def test_word_impact(self):
        change_obj = TopicalChanges(self.roll, samples=50)
        change_obj.word_impact(number=5.0)
        change_obj.word_impact(number=2, fast=False)
        with self.assertRaises(TypeError):
            change_obj.word_impact(number="a")
        with self.assertRaises(TypeError):
            change_obj.word_impact(number=5.5)
        with self.assertRaises(ValueError):
            change_obj.word_impact(number=-5)
        with self.assertRaises(TypeError):
            change_obj.word_impact(date_format=1)
        with self.assertRaises(TypeError):
            change_obj.word_impact(fast="a")

    def test_plot_distances(self):
        change_obj = TopicalChanges(self.roll, samples=50)
        change_obj.plot_distances()
