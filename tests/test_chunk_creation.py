import os
import shutil
import unittest
import numpy as np
import pandas as pd
import sys
sys.path.append('../src')
from ttta.preprocessing.chunk_creation import _get_time_indices

class TestChunkCreation(unittest.TestCase):
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

    def test_time_indices(self):
        _get_time_indices(self.df, how="1D", last_date=pd.to_datetime("2020-12-31"), date_column="date", min_docs_per_chunk=1.0)
        _get_time_indices(self.df, how=[pd.to_datetime("2021-01-01"), pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-03")],
                          last_date=pd.to_datetime("2020-12-31"), date_column="date", min_docs_per_chunk=1.0)

    def test_time_indeces_invalid(self):
        with self.assertRaises(TypeError):
            _get_time_indices(1)
        with self.assertRaises(TypeError):
            _get_time_indices(self.df, 1)
        with self.assertRaises(TypeError):
            _get_time_indices(self.df, last_date=1)
        with self.assertRaises(ValueError):
            _get_time_indices(self.df, last_date=pd.to_datetime("2021-12-31"))
        with self.assertRaises(TypeError):
            _get_time_indices(self.df, date_column=1)
        with self.assertRaises(ValueError):
            _get_time_indices(self.df, min_docs_per_chunk=-1)
        with self.assertRaises(TypeError):
            _get_time_indices(self.df, min_docs_per_chunk=1.5)

