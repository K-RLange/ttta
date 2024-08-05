import os
import shutil
import unittest
import numpy as np
import pandas as pd
import sys
sys.path.append('../src')
from ttta.methods.rolling_lda import RollingLDA
from ttta.methods.lda_prototype import LDAPrototype

class TestRollingLDA(unittest.TestCase):
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
        self.dates = [pd.to_datetime("2021-01-01"), pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-03"),
                      pd.to_datetime("2021-01-04"), pd.to_datetime("2021-01-05"), pd.to_datetime("2021-01-06"),
                      pd.to_datetime("2021-01-07")] * 3
        self.df = pd.DataFrame({"text": self.texts, "date": self.dates})
        self.df.sort_values("date", inplace=True)
        self.roll = RollingLDA(10.0, prototype=5.0, how="1D", warmup=1.0, min_docs_per_chunk=5.0, verbose=2.0,
                          topic_threshold=[3.0, 0.0], initial_epochs=200.0, subsequent_epochs=100.0,
                          memory=2.0, alpha=0.05, gamma=0.05, min_count=1.0, max_assign=True, prototype_measure="cosine")

    def test_init(self):
        roll = RollingLDA(10)
        self.assertEqual(roll._K, 10)
        self.assertEqual(roll._prototype, 10)
        self.assertEqual(roll._threshold, [5, 0.002])
        self.assertEqual(roll._initial_epochs, 100)
        self.assertEqual(roll._subsequent_epochs, 50)
        self.assertEqual(roll._memory, 3)
        self.assertEqual(roll._warmup, 48)
        self.assertEqual(roll._how, "M")
        self.assertEqual(roll._min_docs_per_chunk, 10 * 10)
        self.assertEqual(roll._verbose, 1)
        self.assertEqual(roll._alpha, 0.1)
        self.assertEqual(roll._gamma, 0.1)
        self.assertEqual(roll._min_count, 2)
        self.assertEqual(roll._max_assign, False)
        self.assertEqual(roll._measure, "jaccard")
        self.assertIsInstance(roll.lda, LDAPrototype)
        roll = RollingLDA(10.0, prototype=5.0, how="1D", warmup=1.0, min_docs_per_chunk=5.0, verbose=2.0,
                          topic_threshold=[3.0, 0.0], initial_epochs=200.0, subsequent_epochs=100.0,
                          memory=2.0, alpha=0.05, gamma=0.05, min_count=1.0, max_assign=True, prototype_measure="cosine")

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            roll = RollingLDA(-1)
        with self.assertRaises(TypeError):
            roll = RollingLDA("a")
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, prototype=-3)
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, prototype="a")
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, topic_threshold="a")
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, topic_threshold=["a", 0.1])
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, topic_threshold=[-1, 1.0])
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, topic_threshold=[1, -1.0])
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, topic_threshold=[1, 1.0])
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, initial_epochs="a")
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, initial_epochs=3.5)
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, initial_epochs=-1)
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, subsequent_epochs=-5)
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, subsequent_epochs="a")
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, subsequent_epochs=3.5)
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, memory="a")
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, memory=3.5)
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, memory=-1)
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, warmup=-2)
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, warmup="a")
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, warmup=3.5)
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, how=1)
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, min_docs_per_chunk="a")
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, min_docs_per_chunk=3.5)
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, min_docs_per_chunk=-1)
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, verbose="a")
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, alpha="a")
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, alpha=-1)
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, alpha=np.array([-1.0] * 10))
        with self.assertRaises(TypeError):
            roll = RollingLDA(10, alpha=np.array(["a"] * 10))
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, alpha=np.array([0.5] * 5))

    def test_fit(self):
        roll = RollingLDA(10, min_count=1, topic_threshold=[1, 0.001], prototype=1, how="1D", warmup=2, memory=2, min_docs_per_chunk=1)
        roll.fit(self.df, text_column="text", date_column="date", workers=1.0)
        self.assertIsInstance(roll.chunk_indices, pd.DataFrame)
        self.assertEqual(roll.chunk_indices.shape, (7, 4))
        self.assertTrue(roll.lda.is_trained())
        self.assertEqual(roll._last_text["index"], len(self.texts) - 1)
        self.assertEqual(roll._last_text["date"], self.dates[-1])

    def test_fit_invalid(self):
        roll = RollingLDA(10, min_count=1, topic_threshold=[1, 0.001], prototype=1, how="1D", warmup=2, memory=2, min_docs_per_chunk=1)
        with self.assertRaises(TypeError):
            roll.fit(self.df, text_column=1)
        with self.assertRaises(ValueError):
            roll.fit(self.df, text_column="a")
        with self.assertRaises(TypeError):
            roll.fit(self.df, date_column=1)
        with self.assertRaises(ValueError):
            roll.fit(self.df, date_column="a")
        with self.assertRaises(TypeError):
            roll.fit("a")
        with self.assertRaises(TypeError):
            roll.fit(self.df, text_column="date")
        with self.assertRaises(TypeError):
            roll.fit(self.df, workers=0.5)
        with self.assertRaises(ValueError):
            roll.fit(self.df)
            roll.fit(self.df)
        with self.assertRaises(ValueError):
            roll = RollingLDA(10, min_count=1, min_docs_per_chunk=1, warmup=100)
            roll.fit(self.df)

    def test_get_parameters(self):
        params = self.roll.get_parameters()
        self.assertEqual(params["_K"], 10)
        self.assertEqual(params["_prototype"], 5)
        self.assertEqual(params["_threshold"], [3, 0.0])
        self.assertEqual(params["_initial_epochs"], 200)
        self.assertEqual(params["_subsequent_epochs"], 100)
        self.assertEqual(params["_memory"], 2)
        self.assertEqual(params["_warmup"], 1)
        self.assertEqual(params["_how"], "1D")
        self.assertEqual(params["_min_docs_per_chunk"], 5)
        self.assertEqual(params["_verbose"], 2)
        self.assertEqual(params["_alpha"], 0.05)
        self.assertEqual(params["_gamma"], 0.05)
        self.assertEqual(params["_min_count"], 1)
        self.assertEqual(params["_max_assign"], True)
        self.assertEqual(params["_measure"], "cosine")

    def test_set_parameters(self):
        roll = RollingLDA(10)
        roll.set_parameters({"_K": 5, "_prototype": 3, "_threshold": [2, 0.1], "_initial_epochs": 100, "_subsequent_epochs": 50,
                            "_memory": 3, "_warmup": 2, "_how": "2D", "_min_docs_per_chunk": 10, "_verbose": 1, "_alpha": 0.1,
                            "_gamma": 0.1, "_min_count": 2, "_max_assign": False, "_measure": "jaccard"})
        self.assertEqual(roll._K, 5)
        self.assertEqual(roll._prototype, 3)
        self.assertEqual(roll._threshold, [2, 0.1])
        self.assertEqual(roll._initial_epochs, 100)
        self.assertEqual(roll._subsequent_epochs, 50)
        self.assertEqual(roll._memory, 3)
        self.assertEqual(roll._warmup, 2)
        self.assertEqual(roll._how, "2D")
        self.assertEqual(roll._min_docs_per_chunk, 10)
        self.assertEqual(roll._verbose, 1)
        self.assertEqual(roll._alpha, 0.1)
        self.assertEqual(roll._gamma, 0.1)
        self.assertEqual(roll._min_count, 2)
        self.assertEqual(roll._max_assign, False)
        self.assertEqual(roll._measure, "jaccard")
        with self.assertRaises(TypeError):
            roll.set_parameters(1)

    def test_save_and_load(self):
        self.roll.save("test.pickle")
        roll = RollingLDA(5)
        roll.load("test.pickle")
        self.assertEqual(roll._K, 10)
        self.assertEqual(roll._prototype, 5)
        self.assertEqual(roll._threshold, [3, 0.0])
        self.assertEqual(roll._initial_epochs, 200)
        self.assertEqual(roll._subsequent_epochs, 100)
        self.assertEqual(roll._memory, 2)
        self.assertEqual(roll._warmup, 1)
        self.assertEqual(roll._how, "1D")
        self.assertEqual(roll._min_docs_per_chunk, 5)
        self.assertEqual(roll._verbose, 2)
        self.assertEqual(roll._alpha, 0.05)
        self.assertEqual(roll._gamma, 0.05)
        self.assertEqual(roll._min_count, 1)
        self.assertEqual(roll._max_assign, True)
        self.assertEqual(roll._measure, "cosine")
        os.remove("test.pickle")
        with self.assertRaises(FileNotFoundError):
            roll.load("test.pickle")
        with self.assertRaises(TypeError):
            roll.load(1)
        with self.assertRaises(TypeError):
            roll.save(1)

    def test_fit_update(self):
        roll = RollingLDA(10, min_count=1, topic_threshold=[1, 0.001], prototype=1, how="1D", warmup=1, memory=2, min_docs_per_chunk=1)
        roll.fit(self.df.iloc[:9], text_column="text", date_column="date", workers=1.0)
        roll.fit_update(self.df[9:], text_column="text", date_column="date", workers=1.0)
        self.assertEqual(roll._last_text["index"], len(self.texts) - 1)
        self.assertEqual(roll._last_text["date"], self.dates[-1])
        roll = RollingLDA(10, min_count=1, topic_threshold=[1, 0.001], prototype=1, how="1D", warmup=1, memory=2, min_docs_per_chunk=1)
        roll.fit_update(self.df, text_column="text", date_column="date", workers=1.0)

    def test_fit_update_invalid(self):
        roll = RollingLDA(10, min_count=1, topic_threshold=[1, 0.001], prototype=1, how="1D", warmup=1, memory=2, min_docs_per_chunk=1)
        roll.fit(self.df[:9], text_column="text", date_column="date", workers=1.0)
        with self.assertRaises(TypeError):
            roll.fit_update("a")
        with self.assertRaises(ValueError):
            roll.fit_update(self.df)
        with self.assertRaises(TypeError):
            roll.fit_update(self.df[9:], text_column=1)
        with self.assertRaises(ValueError):
            roll.fit_update(self.df[9:], text_column="a")
        with self.assertRaises(TypeError):
            roll.fit_update(self.df[9:], date_column=1)
        with self.assertRaises(ValueError):
            roll.fit_update(self.df[9:], date_column="a")
        with self.assertRaises(TypeError):
            roll.fit_update(self.df[9:], text_column="date")
        with self.assertRaises(TypeError):
            roll.fit_update(self.df[9:], workers=0.5)
        with self.assertRaises(TypeError):
            roll.fit_update(self.df[9:], how=0.5)
        with self.assertRaises(TypeError):
            alt_df = self.df.copy()
            alt_df["text"] = alt_df["text"].apply(lambda x: " ".join(x))
            roll.fit_update(self.df[9:], how=0.5)

    def test_top_words(self):
        roll = RollingLDA(10, min_count=1, topic_threshold=[1, 0.001], prototype=1, how="1D", warmup=1, memory=2, min_docs_per_chunk=1)
        roll.fit(self.df, text_column="text", date_column="date", workers=1.0)
        roll.top_words(0.0, number=5.0, topic=1.0, return_as_data_frame=False, )
        self.assertIsInstance(roll.top_words(0, return_as_data_frame=False), list)
        self.assertEqual(len(roll.top_words(0, return_as_data_frame=False)), 10)
        self.assertEqual(len(roll.top_words(0, return_as_data_frame=False)[0]), 5)
        self.assertIsInstance(roll.top_words(return_as_data_frame=False), list)
        self.assertEqual(len(roll.top_words(return_as_data_frame=False)), 10)
        self.assertEqual(len(roll.top_words(return_as_data_frame=False)[0]), 5)
        self.assertIsInstance(roll.top_words(0, return_as_data_frame=True), pd.DataFrame)
        self.assertIsInstance(roll.top_words("all", return_as_data_frame=True), pd.DataFrame)
        self.assertIsInstance(roll.top_words(return_as_data_frame=True), pd.DataFrame)

    def test_top_words_invalid(self):
        roll = RollingLDA(10, min_count=1, topic_threshold=[1, 0.001], prototype=1, how="1D", warmup=1, memory=2, min_docs_per_chunk=1)
        roll.fit(self.df, text_column="text", date_column="date", workers=1.0)
        with self.assertRaises(TypeError):
            roll.top_words(0, return_as_data_frame="a")
        with self.assertRaises(TypeError):
            roll.top_words(0.5)
        with self.assertRaises(ValueError):
            roll.top_words(0, number=0)
        with self.assertRaises(TypeError):
            roll.top_words(0, number=3.5)
        with self.assertRaises(TypeError):
            roll.top_words(0, topic=3.5)
        with self.assertRaises(TypeError):
            roll.top_words("a")
        with self.assertRaises(TypeError):
            roll.top_words(1, importance="a")

    def test_get_word_topic_matrix(self):
        roll = RollingLDA(10, min_count=1, topic_threshold=[1, 0.001], prototype=1, how="1D", warmup=1, memory=2, min_docs_per_chunk=1)
        roll.fit(self.df, text_column="text", date_column="date", workers=1.0)
        with self.assertRaises(TypeError):
            roll.get_word_topic_matrix("a")
        roll.get_word_topic_matrix(0)

    def test_get_time_indices(self):
        roll = RollingLDA(10, min_count=1, topic_threshold=[1, 0.001], prototype=1, how=[pd.to_datetime("2021-01-01"), pd.to_datetime("2021-01-02")], warmup=0, memory=1, min_docs_per_chunk=1)
        roll.fit(self.df[:6], text_column="text", date_column="date", workers=1.0)
        roll._get_time_indices(self.df[6:], how="1D")
        with self.assertRaises(TypeError):
            roll._get_time_indices(1)
        with self.assertRaises(TypeError):
            roll._get_time_indices(self.df[6:], update=1)
        roll = RollingLDA(10, min_count=1, topic_threshold=[1, 0.001], prototype=1, how="1D", warmup=0, memory=1, min_docs_per_chunk=1)
        roll.fit(self.df[:6], text_column="text", date_column="date", workers=1.0)
        roll._get_time_indices(self.df[6:], how="2D")
        roll._get_time_indices(self.df[6:], how=self.dates[3:7])

    def test_wordclouds(self):
        roll = RollingLDA(3, min_count=1, topic_threshold=[1, 0.001], prototype=1, how="1D", warmup=1, memory=2, min_docs_per_chunk=1)
        roll.fit(self.df, text_column="text", date_column="date", workers=1.0)
        roll.wordclouds()
        with self.assertRaises(TypeError):
            roll.wordclouds(0)
        with self.assertRaises(TypeError):
            roll.wordclouds(path=1)
        with self.assertRaises(TypeError):
            roll.wordclouds(topic=1.5)
        with self.assertRaises(TypeError):
            roll.wordclouds(number=1.5)
        with self.assertRaises(TypeError):
            roll.wordclouds(height=1.5)
        with self.assertRaises(TypeError):
            roll.wordclouds(width=1.5)
        with self.assertRaises(TypeError):
            roll.wordclouds(show=1.5)
        with self.assertRaises(TypeError):
            roll.wordclouds([0.5])