import os
import shutil
import unittest
import numpy as np
import pandas as pd
import sys
sys.path.append('../src')
from ttta.methods.topic_matching import TopicClusters
from ttta.methods.lda_prototype import LDAPrototype

class TestTopicMatching(unittest.TestCase):
    def setUp(self):
        self.texts = [["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"],
                      ["sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore", "magna"],
                      ["aliqua", "ut", "enim", "ad", "minim", "veniam", "quis", "nostrud", "exercitation", "ullamco"],
                      ["laboris", "nisi", "ut", "aliquip", "ex", "ea", "commodo", "consequat"],
                      ["duis", "aute", "irure", "dolor", "in", "reprehenderit", "in", "voluptate", "velit", "esse"],
                      ["cillum", "dolore", "eu", "fugiat", "nulla", "pariatur"],
                      ["excepteur", "sint", "occaecat", "cupidatat", "non", "proident", "sunt", "in", "culpa", "qui"],
                      ["officia", "deserunt", "mollit", "anim", "id", "est", "laborum"]]
        self.lda = LDAPrototype(K=3, prototype=1, min_count=0, verbose=2, max_assign=True)
        self.lda.fit(self.texts, epochs=10.0, workers=1.0, first_chunk=True, chunk_end=float(len(self.texts)), memory_start=1.0)
        self.matrices = [self.lda.get_word_topic_matrix(), self.lda.get_word_topic_matrix()]

    def test_init(self):
        model = TopicClusters(self.matrices, K=3.0, topic_threshold=[1, 0.01])
        model.get_matched_topics()
        with self.assertRaises(TypeError):
            TopicClusters(1)
        with self.assertRaises(ValueError):
            TopicClusters([self.lda.get_word_topic_matrix()])
        with self.assertRaises(TypeError):
            TopicClusters([1])
        with self.assertRaises(ValueError):
            TopicClusters(self.matrices, measure="a")
        with self.assertRaises(TypeError):
            TopicClusters(self.matrices, measure=1)
        with self.assertRaises(ValueError):
            TopicClusters(self.matrices, K=1)
        with self.assertRaises(ValueError):
            TopicClusters(self.matrices, K=4)
        with self.assertRaises(TypeError):
            TopicClusters(self.matrices, K=3.5)
        with self.assertRaises(TypeError):
            TopicClusters(self.matrices, topic_threshold=1)
        with self.assertRaises(ValueError):
            TopicClusters(self.matrices, topic_threshold=[-1, 0.1])
        with self.assertRaises(ValueError):
            TopicClusters(self.matrices, topic_threshold=[1, -0.1])
        with self.assertRaises(ValueError):
            TopicClusters(self.matrices, topic_threshold=[1, 1.1])

    def test_cluster_topics(self):
        model = TopicClusters(self.matrices, K=3.0, topic_threshold=[1, 0.01])
        with self.assertRaises(TypeError):
            model._cluster_topics("a", 1, "cosine", dict())
        with self.assertRaises(TypeError):
            model._cluster_topics(1.5, 1, "cosine", dict())
        with self.assertRaises(TypeError):
            model._cluster_topics(1, "a", "cosine", dict())
        with self.assertRaises(TypeError):
            model._cluster_topics(1, 1.5, "cosine", dict())
        with self.assertRaises(TypeError):
            model._cluster_topics(1, 1, 1, dict())
        with self.assertRaises(TypeError):
            model._cluster_topics(1, 1, "cosine", [])