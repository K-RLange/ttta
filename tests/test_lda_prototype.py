import os
import unittest
import numpy as np
import pandas as pd
import sys
sys.path.append('../src')
from ttta.methods.lda_prototype import LDAPrototype

class TestLDAPrototype(unittest.TestCase):

    def setUp(self):
        self.texts = [["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"],
                      ["sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore", "magna"],
                      ["aliqua", "ut", "enim", "ad", "minim", "veniam", "quis", "nostrud", "exercitation", "ullamco"],
                      ["laboris", "nisi", "ut", "aliquip", "ex", "ea", "commodo", "consequat"],
                      ["duis", "aute", "irure", "dolor", "in", "reprehenderit", "in", "voluptate", "velit", "esse"],
                      ["cillum", "dolore", "eu", "fugiat", "nulla", "pariatur"],
                      ["excepteur", "sint", "occaecat", "cupidatat", "non", "proident", "sunt", "in", "culpa", "qui"],
                      ["officia", "deserunt", "mollit", "anim", "id", "est", "laborum"]]
        self.total_words = len([word for doc in self.texts for word in doc])
        self.unique_words = len(set([word for doc in self.texts for word in doc]))
    def test_init_with_valid_parameters(self):
        lda = LDAPrototype(K=5.0, alpha=0.2, gamma=0.2, prototype=3.0, topic_threshold=[5.0, 0.002], prototype_measure="jaccard",
                           min_count=2.0, max_assign=False, verbose=1)
        self.assertIsInstance(lda, LDAPrototype)
        self.assertEqual(lda._K, 5)
        self.assertEqual(lda._alpha.shape[0], 5)
        self.assertEqual(lda._gamma.shape[0], 5)
        self.assertEqual(lda._prototype, 3)
        self.assertEqual(lda._threshold, [5, 0.002])
        self.assertEqual(lda._measure, "jaccard")
        self.assertEqual(lda._min_count, 2)
        self.assertFalse(lda._max_assign)
        self.assertEqual(lda._verbose, 1)

    def test_init_invalid_parameters(self):
        with self.assertRaises(TypeError):
            lda = LDAPrototype()
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=0)
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K="K")
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=1.5)
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, alpha=0)
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, alpha="alpha")
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, alpha=-0.5)
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, alpha=np.array([1, 1, 1]))
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, alpha=np.array([-1, 1, 1, 1, 1]))
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, gamma=-0.5)
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, gamma=np.array([1, 1, 1]))
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, gamma=np.array([-1, 1, 1, 1, 1]))
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, gamma=0)
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, gamma="gamma")
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, prototype=0)
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, prototype="prototype")
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, prototype=1.5)
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, topic_threshold=0)
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, topic_threshold=[0.1, 0.1])
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, topic_threshold=[1, "a"])
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, topic_threshold=[1, 1.0])
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, prototype_measure=0)
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, prototype_measure="prototype_measure")
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, min_count="min_count")
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, min_count=3.5)
        with self.assertRaises(ValueError):
            lda = LDAPrototype(K=5, min_count=-1)
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, max_assign=3)
        with self.assertRaises(TypeError):
            lda = LDAPrototype(K=5, verbose="verbose")

    def test_fit_with_valid_input(self):
        lda = LDAPrototype(K=3, prototype=1, min_count=0, verbose=2, max_assign=True)
        lda.fit([[0, 1, 2, 3], [4, 5, 6, 7, 8]], epochs=10.0, workers=1.0, first_chunk=True, chunk_end=float(len(self.texts)), memory_start=1.0)
        self.assertTrue(lda._is_trained)
        self.assertIsNotNone(lda._assignments)
        self.assertIsNotNone(lda._word_vec)
        self.assertIsNotNone(lda._document_topic_matrix)

    def test_fit_invalid_input(self):
        lda = LDAPrototype(K=3, prototype=3, min_count=0)
        with self.assertRaises(TypeError):
            lda.fit(0, epochs=10)
        with self.assertRaises(TypeError):
            lda.fit(np.array([0]), epochs=10)
        with self.assertRaises(TypeError):
            lda.fit("test", epochs=10)
        with self.assertRaises(TypeError):
            lda.fit([0, 0], epochs=10)
        with self.assertRaises(TypeError):
            lda.fit(self.texts, epochs="epochs")
        with self.assertRaises(ValueError):
            lda.fit(self.texts, epochs=-1)
        with self.assertRaises(TypeError):
            lda.fit(self.texts, epochs=1.5)
        with self.assertRaises(TypeError):
            lda.fit(self.texts, first_chunk=-1)
        with self.assertRaises(TypeError):
            lda.fit(self.texts, chunk_end="chunk_end")
        with self.assertRaises(ValueError):
            lda.fit(self.texts, chunk_end=-1)
        with self.assertRaises(TypeError):
            lda.fit(self.texts, chunk_end=1.5)
        with self.assertRaises(TypeError):
            lda.fit(self.texts, memory_start="memory_start")
        with self.assertRaises(ValueError):
            lda.fit(self.texts, memory_start=-1)
        with self.assertRaises(TypeError):
            lda.fit(self.texts, memory_start=1.5)
        with self.assertRaises(TypeError):
            lda.fit(self.texts, workers="workers")
        with self.assertRaises(TypeError):
            lda.fit(self.texts, workers=1.5)
    def test_top_words(self):
        lda = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        lda2 = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        lda.fit(self.texts, epochs=10)
        lda2.fit(self.texts, epochs=10)
        top_words = lda.top_words(topic=1.0, number=2.0, importance=False)
        self.assertIsInstance(top_words, list)
        self.assertEqual(len(top_words), 2)
        top_words = lda.top_words(number=2.0, return_as_data_frame=False)
        self.assertIsInstance(top_words, list)
        self.assertEqual(len(top_words), 3)
        self.assertIsInstance(top_words[0], list)
        self.assertEqual(len(top_words[0]), 2)
        top_words = lda.top_words(number=2)
        top_words2 = lda2.top_words(number=2)
        self.assertIsInstance(top_words, pd.DataFrame)
        self.assertEqual(len(top_words), 2)
        self.assertEqual(len(top_words.columns), 3)
        self.assertFalse(top_words.equals(top_words2))
        top_words = lda.top_words(topic=1.0, number=2.0, importance=False)

    def test_top_words_invalid_input(self):
        lda = LDAPrototype(K=3, prototype=3, min_count=0, topic_threshold=[1, 0.002])
        lda2 = LDAPrototype(K=3, prototype=3, min_count=0, topic_threshold=[1, 0.002])
        lda.fit(self.texts, epochs=10)
        with self.assertRaises(TypeError):
            top_words = lda.top_words(topic=3.5)
        with self.assertRaises(ValueError):
            top_words = lda.top_words(number=0)
        with self.assertRaises(TypeError):
            top_words = lda.top_words(number="number")
        with self.assertRaises(TypeError):
            top_words = lda.top_words(number=3.5)
        with self.assertRaises(TypeError):
            top_words = lda.top_words(topic="topic")
        with self.assertRaises(ValueError):
            top_words = lda.top_words(topic=0)
        with self.assertRaises(ValueError):
            top_words = lda.top_words(topic=4)
        with self.assertRaises(TypeError):
            top_words = lda.top_words(importance="importance")
        with self.assertRaises(TypeError):
            top_words = lda.top_words(word_topic_matrix="word_topic_matrix")
        with self.assertRaises(TypeError):
            top_words = lda.top_words(return_as_data_frame=-1)
        with self.assertRaises(AttributeError):
            top_words = lda2.top_words()

    def test_get_word_and_doc_vector(self):
        lda = LDAPrototype(K=3, prototype=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        word_vec, doc_vec = lda._get_word_and_doc_vector(lda._dtm)
        self.assertIsInstance(word_vec, np.ndarray)
        self.assertIsInstance(doc_vec, np.ndarray)
        self.assertEqual(word_vec.shape, doc_vec.shape)

    def test_get_word_and_doc_vector_invalid_input(self):
        lda = LDAPrototype(K=3, prototype=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        with self.assertRaises(TypeError):
            word_vec, doc_vec = lda._get_word_and_doc_vector(0)
        with self.assertRaises(TypeError):
            word_vec, doc_vec = lda._get_word_and_doc_vector()

    def test_get_word_topic_matrix(self):
        lda = LDAPrototype(K=3, prototype=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        word_topic_matrix = lda.get_word_topic_matrix()
        self.assertIsInstance(word_topic_matrix, np.ndarray)
        self.assertEqual(word_topic_matrix.shape, (self.unique_words, 3))

    def test_get_word_topic_matrix_invalid_input(self):
        lda = LDAPrototype(K=3, prototype=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        with self.assertRaises(ValueError):
            word_topic_matrix = lda.get_word_topic_matrix(np.ndarray([0]), lda.get_assignment_vec())
        with self.assertRaises(ValueError):
            word_topic_matrix = lda.get_word_topic_matrix(0, lda.get_assignment_vec())
        with self.assertRaises(TypeError):
            word_topic_matrix = lda.get_word_topic_matrix("a", lda.get_assignment_vec())
        with self.assertRaises(TypeError):
            word_topic_matrix = lda.get_word_topic_matrix(lda.get_word_vec(), "a")

    def test_get_assignment_vec(self):
        lda = LDAPrototype(K=3, prototype=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        assignment_vec = lda.get_assignment_vec()
        self.assertIsInstance(assignment_vec, np.ndarray)
        self.assertEqual(assignment_vec.shape, (self.total_words,))

    def test_get_word_vec(self):
        lda = LDAPrototype(K=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        word_vec = lda.get_word_vec()
        self.assertIsInstance(word_vec, np.ndarray)
        self.assertEqual(word_vec.shape, (self.total_words,))

    def test_get_doc_vec(self):
        lda = LDAPrototype(K=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        doc_vec = lda.get_doc_vec()
        self.assertIsInstance(doc_vec, np.ndarray)
        self.assertEqual(doc_vec.shape, (self.total_words,))
    def test_get_document_topic_matrix(self):
        lda = LDAPrototype(K=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        document_topic_matrix = lda.get_document_topic_matrix(lda.get_doc_vec(), lda.get_assignment_vec())
        self.assertIsInstance(document_topic_matrix, np.ndarray)
        self.assertEqual(document_topic_matrix.shape, (len(self.texts), 3))

    def test_get_document_topic_matrix_invalid_input(self):
        lda = LDAPrototype(K=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        with self.assertRaises(ValueError):
            document_topic_matrix = lda.get_document_topic_matrix(0, lda.get_assignment_vec())
        with self.assertRaises(TypeError):
            document_topic_matrix = lda.get_document_topic_matrix(lda.get_doc_vec(), "a")
        with self.assertRaises(TypeError):
            document_topic_matrix = lda.get_document_topic_matrix("a", lda.get_assignment_vec())

    def test_get_vocab(self):
        lda = LDAPrototype(K=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        vocab = lda.get_vocab()
        self.assertIsInstance(vocab, list)
        self.assertEqual(len(vocab), self.unique_words)

    def test_get_params(self):
        lda = LDAPrototype(K=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        params = lda.get_params()
        self.assertIsInstance(params, dict)
        self.assertEqual(params["_K"], 3)
        self.assertTrue(np.array_equal(params["_alpha"], [1/3] * 3))
        self.assertTrue(np.array_equal(params["_gamma"], [1/3] * 3))
        self.assertEqual(params["_prototype"], 10)
        self.assertEqual(params["_threshold"], [5, 0.002])
        self.assertEqual(params["_measure"], "jaccard")
        self.assertEqual(params["_min_count"], 0)
        self.assertFalse(params["_max_assign"], False)
        self.assertEqual(params["_verbose"], 1)

    def test_set_params(self):
        lda = LDAPrototype(K=3, min_count=0)
        lda.fit(self.texts, epochs=10)
        lda.set_params({"_K": 5, "_alpha": np.array([0.5] * 5), "_gamma": np.array([0.5] * 5), "_prototype": 2, "_threshold": [3, 0.002], "_measure": "cosine",
                        "_min_count": 3, "_max_assign": True, "_verbose": 0})
        self.assertEqual(lda._K, 5)
        self.assertTrue(np.array_equal(lda._alpha, [0.5] * 5))
        self.assertTrue(np.array_equal(lda._gamma, [0.5] * 5))
        self.assertEqual(lda._prototype, 2)
        self.assertEqual(lda._threshold, [3, 0.002])
        self.assertEqual(lda._measure, "cosine")
        self.assertEqual(lda._min_count, 3)
        self.assertTrue(lda._max_assign)
        self.assertEqual(lda._verbose, 0)
        with self.assertRaises(TypeError):
            lda.set_params(0)

    def test_save_and_load(self):
        lda = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        lda.fit(self.texts, epochs=10)
        lda.save("lda_test.pickle")
        lda2 = LDAPrototype(K=3, min_count=0)
        lda2.load("lda_test.pickle")
        os.remove("lda_test.pickle")

    def test_save_and_load_invalid_input(self):
        lda = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        lda.fit(self.texts, epochs=10)
        with self.assertRaises(TypeError):
            lda.save(0)
        with self.assertRaises(TypeError):
            lda.load(0)

    def test_shrink_model(self):
        lda = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        with self.assertRaises(NotImplementedError):
            lda.shrink_model(2)

    def test_is_trained(self):
        lda = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        self.assertFalse(lda.is_trained())
        lda.fit(self.texts, epochs=10)
        self.assertTrue(lda.is_trained())

    def test_create_lda_parameters(self):
        lda = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        self.assertCountEqual(lda._create_lda_parameters(0.5), [0.5, 0.5, 0.5])
        self.assertCountEqual(lda._create_lda_parameters(np.array([1, 1, 1])), [1, 1, 1])

    def test_create_lda_parameters_invalid_input(self):
        lda = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        with self.assertRaises(TypeError):
            lda._create_lda_parameters("a")
        with self.assertRaises(ValueError):
            lda._create_lda_parameters(-1)
        with self.assertRaises(ValueError):
            lda._create_lda_parameters([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            lda._create_lda_parameters(np.array([-1, 2, 4]))

    def test_wordclouds(self):
        lda = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        lda.fit(self.texts, epochs=10)
        lda.wordclouds(None, 1.0, height=100.0, width=100.0, show=False)
        lda.wordclouds(1.0, 5.0, None, height=100.0, width=100.0, show=True, word_topic_matrix=lda.get_word_topic_matrix(lda.get_word_vec(), lda.get_assignment_vec()))
        os.remove("wordclouds.pdf")

    def test_wordclouds_invalid_input(self):
        lda = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        lda.fit(self.texts, epochs=10)
        with self.assertRaises(TypeError):
            lda.wordclouds(1.5)
        with self.assertRaises(ValueError):
            lda.wordclouds(-1)
        with self.assertRaises(ValueError):
            lda.wordclouds(number=-1)
        with self.assertRaises(TypeError):
            lda.wordclouds(number=0.5)
        with self.assertRaises(TypeError):
            lda.wordclouds(path=0.5)
        with self.assertRaises(ValueError):
            lda.wordclouds(width=-1)
        with self.assertRaises(TypeError):
            lda.wordclouds(width=0.5)
        with self.assertRaises(ValueError):
            lda.wordclouds(height=-1)
        with self.assertRaises(TypeError):
            lda.wordclouds(height=0.5)
        with self.assertRaises(TypeError):
            lda.wordclouds(show=0.5)
        with self.assertRaises(ValueError):
            lda.wordclouds(path=None, show=False)

    def test_calculate_prototype_invalid_input(self):
        lda = LDAPrototype(K=3, min_count=0, prototype=3, topic_threshold=[1, 0.002])
        lda.fit(self.texts, epochs=10)
        with self.assertRaises(TypeError):
            lda._calculate_prototype(0.5, lda.get_doc_vec(), lda.get_word_topic_matrix(), lda.get_document_topic_matrix(lda.get_doc_vec(), lda.get_assignment_vec()), np.array(1), 1, 10, 0)
        with self.assertRaises(TypeError):
            lda._calculate_prototype(lda.get_word_vec(), 0, lda.get_word_topic_matrix(), lda.get_document_topic_matrix(lda.get_doc_vec(), lda.get_assignment_vec()), np.array(1), 1, 10, 0)
        with self.assertRaises(TypeError):
            lda._calculate_prototype(lda.get_word_vec(), lda.get_doc_vec(), 0, lda.get_document_topic_matrix(lda.get_doc_vec(), lda.get_assignment_vec()), np.array(1), 1, 10, 0)
        with self.assertRaises(TypeError):
            lda._calculate_prototype(lda.get_word_vec(), lda.get_doc_vec(), lda.get_word_topic_matrix(), 0, np.array(1), 1, 10, 0)
        with self.assertRaises(TypeError):
            lda._calculate_prototype(lda.get_word_vec(), lda.get_doc_vec(), lda.get_word_topic_matrix(), lda.get_document_topic_matrix(lda.get_doc_vec(), lda.get_assignment_vec()), 0.5, 1, 10, 0)
        with self.assertRaises(TypeError):
            lda._calculate_prototype(lda.get_word_vec(), lda.get_doc_vec(), lda.get_word_topic_matrix(), lda.get_document_topic_matrix(lda.get_doc_vec(), lda.get_assignment_vec()), np.array(1), 1.5, 10, 0)
        with self.assertRaises(TypeError):
            lda._calculate_prototype(lda.get_word_vec(), lda.get_doc_vec(), lda.get_word_topic_matrix(), lda.get_document_topic_matrix(lda.get_doc_vec(), lda.get_assignment_vec()), np.array(1), 1, 10.5, 0)
        with self.assertRaises(TypeError):
            lda._calculate_prototype(lda.get_word_vec(), lda.get_doc_vec(), lda.get_word_topic_matrix(), lda.get_document_topic_matrix(lda.get_doc_vec(), lda.get_assignment_vec()), np.array(1), 1, 10, 0.5)
        with self.assertRaises(ValueError):
            lda._calculate_prototype(lda.get_word_vec(), lda.get_doc_vec(), lda.get_word_topic_matrix(), lda.get_document_topic_matrix(lda.get_doc_vec(), lda.get_assignment_vec()), np.array(1), 1, 10, 0-1)
if __name__ == '__main__':
    unittest.main()
