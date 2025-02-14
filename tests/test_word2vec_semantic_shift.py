import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import pickle
from datetime import datetime
import matplotlib
# Use a non-interactive backend for testing visualization.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from src.ttta.methods.word2vec_semantic_shift import Word2VecSemanticShift

# The Word2VecSemanticShift class uses _get_time_indices from the package
# ttta.preprocessing.chunk_creation. For testing we override it with a dummy
# function that always returns one “chunk” covering all texts.
def dummy_get_time_indices(texts, how, date_column, min_docs_per_chunk):
    # Return a DataFrame with a single chunk:
    #   - chunk_start is 0 (i.e. training starts at the beginning)
    #   - Date is taken from the first (converted) date in the texts.
    return pd.DataFrame({
        "chunk_start": [0],
        "Date": [pd.to_datetime(texts[date_column].iloc[0])]
        })


class TestWord2VecSemanticShift(unittest.TestCase):
    def setUp(self):
        """
        Setup runs before each test.
        We:
          - Monkey-patch the _get_time_indices function so that training
            does not depend on external code.
          - Create a dummy DataFrame with a small corpus (each row is a tokenized document)
          - Instantiate the Word2VecSemanticShift model (with verbose=0).
        """
        # Monkey-patch the _get_time_indices function in the module where it was imported.
        # (Assuming the module is "ttta.preprocessing.chunk_creation".)
        import ttta.preprocessing.chunk_creation as chunk_creation
        self.original_get_time_indices = chunk_creation._get_time_indices
        chunk_creation._get_time_indices = dummy_get_time_indices

        # Create dummy data:
        # The texts are tokenized (lists of strings) and dates are provided.
        # With four texts, each word appears at least twice (to overcome the default min_count=2).
        self.df = pd.DataFrame({
            "text": [
                ["hello", "world"],
                ["hello", "python"],
                ["world", "python"],
                ["hello", "world", "python"],
                ["this", "world", "python", "is", "a", "test"],
                ["hello", "python"],
                ["test", "world", "python"],
                ["is", "a", "hello"]
                ],
            "date": ["2022-01-01", "2022-01-01", "2022-01-02", "2022-01-02",
                     "2022-01-03", "2022-01-03", "2022-01-04", "2022-01-04"]
            })

        # Create an instance of the model with verbose=0 (to avoid progress bars).
        self.model = Word2VecSemanticShift(verbose=0, min_count=1, how="1D")

    def tearDown(self):
        """Restore the original _get_time_indices function."""
        import ttta.preprocessing.chunk_creation as chunk_creation
        chunk_creation._get_time_indices = self.original_get_time_indices

    def test_is_trained_before_and_after_fit(self):
        """Test that is_trained() returns False before and True after training."""
        self.assertFalse(self.model.is_trained(),
                         "Model should not be trained initially.")
        self.model.fit(self.df, text_column="text", date_column="date",
                       date_format="%Y-%m-%d")
        self.assertTrue(self.model.is_trained(),
                        "Model should be trained after calling fit().")
        with self.assertRaises(ValueError):
            self.model.fit(self.df, text_column="text", date_column="date",
                           date_format="%Y-%m-%d")

    def test_fit_creates_models(self):
        """Test that fit() creates nonempty Word2Vecs and aligned_models lists."""
        with self.assertRaises(ValueError):
            self.model.fit_update(self.df, text_column="text", date_column="date",
                                  date_format="%Y-%m-%d")
        self.model.fit(self.df, text_column="text", date_column="date",
                       date_format="%Y-%m-%d")
        self.assertGreater(len(self.model.word2vecs), 0,
                           "word2vecs should not be empty after fit().")
        self.assertGreater(len(self.model.aligned_models), 0,
                           "aligned_models should not be empty after fit().")
        self.assertEqual(self.model._last_text, len(self.df),
                         "_last_text should equal number of texts.")

    def test_get_vector_and_infer_vector(self):
        """Test that get_vector() and infer_vector() return vectors of the expected shape."""
        self.model.fit(self.df, text_column="text", date_column="date",
                       date_format="%Y-%m-%d")
        # "hello" should be in the vocabulary.
        vector_direct = self.model.get_vector("hello", 0, aligned=True)
        vector_non_aligned = self.model.get_vector("hello", 0, aligned=False)
        vector_infer = self.model.infer_vector("hello", 0, aligned=True)
        expected_dim = self.model.trainer_args["vector_size"]
        self.assertEqual(vector_direct.shape, (expected_dim,),
                         "Vector shape should match vector_size.")
        self.assertEqual(vector_infer.shape, (expected_dim,),
                         "Inferred vector shape should match vector_size.")

        # Requesting a word not in the vocabulary should raise a ValueError.
        with self.assertRaises(KeyError):
            self.model.get_vector("nonexistent_word", 0, aligned=True)

    def test_top_words(self):
        """Test that top_words() returns a tuple of lists with the expected length."""
        self.model.fit(self.df, text_column="text", date_column="date",
                       date_format="%Y-%m-%d")
        top_words, similarities = self.model.top_words("hello", 0, k=2,
                                                       pos_tag=False,
                                                       aligned=True)
        self.assertIsInstance(top_words, list, "top_words should be a list.")
        self.assertIsInstance(similarities, list,
                              "similarities should be a list.")
        self.assertEqual(len(top_words), 2,
                         "Should return exactly k top words.")
        self.assertEqual(len(similarities), 2,
                         "Should return exactly k similarity scores.")

        # Using a word not in the vocabulary should raise an error.
        with self.assertRaises(ValueError):
            self.model.top_words("nonexistent_word", 0, k=2, pos_tag=False,
                                 aligned=True)

    def test_get_vocab(self):
        """Test that get_vocab() returns a list and raises errors on invalid chunk arguments."""
        self.model.fit(self.df, text_column="text", date_column="date",
                       date_format="%Y-%m-%d")
        vocab_reference = self.model.get_vocab("reference")
        vocab_chunk = self.model.get_vocab(0)
        self.assertIsInstance(vocab_reference, list,
                              "Vocabulary should be a list.")
        self.assertIsInstance(vocab_chunk, list,
                              "Vocabulary should be a list.")

        with self.assertRaises(ValueError):
            self.model.get_vocab(3.14)  # invalid chunk argument type

    def test_get_parameters(self):
        """Test that get_parameters() returns a dictionary that includes 'trainer_args'."""
        params = self.model.get_parameters()
        self.assertIsInstance(params, dict,
                              "Parameters should be returned as a dictionary.")
        self.assertIn("trainer_args", params,
                      "Parameters should include 'trainer_args'.")

    def test_save_and_load(self):
        """Test that save() and load() work as expected."""
        self.model.fit(self.df, text_column="text", date_column="date",
                       date_format="%Y-%m-%d")
        with self.assertRaises(TypeError):
            self.model.save(["not_a_string"])
        with self.assertRaises(TypeError):
            self.model.load(["not_a_string"])
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()  # Close the file so that pickle can open it on Windows.
        try:
            self.model.save(temp_file.name)
            # Create a new instance and load from the saved file.
            new_model = Word2VecSemanticShift()
            new_model.load(temp_file.name)
            self.assertEqual(new_model._last_text, self.model._last_text,
                             "Loaded model should have the same _last_text.")
            self.assertEqual(new_model.get_reference(),
                             self.model.get_reference(),
                             "Loaded model should have the same reference.")
        finally:
            os.remove(temp_file.name)

    def test__prepare_for_training(self):
        """Test that _prepare_for_training() returns a DataFrame with the expected columns."""
        with self.assertRaises(TypeError):
            self.model._prepare_for_training(self.df, ["text"], "date",
                                             "%Y-%m-%d")
        with self.assertRaises(ValueError):
            self.model._prepare_for_training(self.df, "Text", "date",
                                             "%Y-%m-%d")
        with self.assertRaises(TypeError):
            self.model._prepare_for_training(self.df, "text", ["date"],
                                             "%Y-%m-%d")
        with self.assertRaises(ValueError):
            self.model._prepare_for_training(self.df, "text", "Date",
                                             "%Y-%m-%d")
        with self.assertRaises(TypeError):
            self.model._prepare_for_training(self.df, "text", "date",
                                             "%Y-%m-%d", update="yes")

    def test_fit_update(self):
        """Test that fit_update() updates an already trained model and raises errors for unknown dates."""
        # First, train the model.
        self.model._verbose = 1
        self.model.fit(self.df, text_column="text", date_column="date",
                       date_format="%Y-%m-%d")
        # Create update data with a date that exists in the training chunks.
        df_update = pd.DataFrame({
            "text": [["hello", "update"], ["world", "update"], ["python", "update"],
                     ["hello", "world", "python", "update"], ["is", "a", "python"],
                     ["test", "update", "world"]],
            "date": ["2022-01-05", "2022-01-05", "2022-01-06", "2022-01-06",
                     "2022-01-07", "2022-01-07"]
            })
        # This update should succeed without errors.
        self.model.fit_update(df_update, text_column="text",
                              date_column="date", date_format="%Y-%m-%d")
        # Now try to update with a date not in the training chunks.
        df_invalid = pd.DataFrame({
            "text": [["new", "data"]],
            "date": ["2023-01-01"]
            })
        with self.assertRaises(ValueError):
            self.model.fit_update(df_invalid, text_column="text",
                                  date_column="date",
                                  date_format="%Y-%m-%d")

    def test_visualize(self):
        """Test that visualize() runs without error (monkey-patching plt.show and plt.savefig)."""
        self.model.fit(self.df, text_column="text", date_column="date",
                       date_format="%Y-%m-%d")
        # Use the chunk date from chunk_indices for the reference.
        reference_dates = [self.model.chunk_indices["date"].iloc[0], # first chunk
                           self.model.chunk_indices["date"].iloc[-1]] # last chunk

        # Override plt.show and plt.savefig to avoid opening windows or writing files during tests.
        original_show = plt.show
        original_savefig = plt.savefig
        plt.show = lambda: None
        plt.savefig = lambda filename: None
        try:
            self.model.visualize(
                main_word="hello",
                chunks_tocompare=reference_dates,
                reference=0,
                k=2,
                pos_tag=False,
                aligned=True,
                tsne_perplexity=1
                # Use a small perplexity since there are few samples.
                )
            self.model.visualize(
                main_word="hello",
                chunks_tocompare=reference_dates,
                reference=0,
                k=1,
                pos_tag=True,
                aligned=True,
                tsne_perplexity=1
                # Use a small perplexity since there are few samples.
                )
            self.model.visualize(
                main_word="hello",
                ignore_words=["python"],
                chunks_tocompare=reference_dates,
                reference=0,
                k=1,
                aligned=True,
                tsne_perplexity=1
                # Use a small perplexity since there are few samples.
                )
        except Exception as e:
            self.fail(f"visualize() raised an exception: {e}")
        finally:
            plt.show = original_show
            plt.savefig = original_savefig
        with self.assertRaises(ValueError):
            self.model.visualize(
                main_word="hello",
                chunks_tocompare=reference_dates,
                reference=0,
                k=2,
                pos_tag=False,
                aligned=True,
                tsne_perplexity=1000
                # Use a small perplexity since there are few samples.
                )
        with self.assertRaises(ValueError):
            self.model.visualize(
                main_word="notinthevocabulary",
                chunks_tocompare=reference_dates,
                reference=0,
                k=2,
                pos_tag=False,
                aligned=True,
                tsne_perplexity=1,
                )

    def test_prepare_for_training_errors(self):
        """Test that _prepare_for_training() raises appropriate errors for invalid input."""
        # Passing a non-DataFrame should raise TypeError.
        with self.assertRaises(TypeError):
            self.model._prepare_for_training("not a dataframe", "text",
                                             "date", "%Y-%m-%d")

        # DataFrame missing the specified text column.
        df_missing = pd.DataFrame({
            "wrong_text": [["hello"]],
            "date": ["2022-01-01"]
            })
        with self.assertRaises(ValueError):
            self.model._prepare_for_training(df_missing, "text", "date",
                                             "%Y-%m-%d")

        # DataFrame with text column elements not being lists (should be tokenized lists).
        df_wrong_type = pd.DataFrame({
            "text": ["not a list"],
            "date": ["2022-01-01"]
            })
        with self.assertRaises(TypeError):
            self.model._prepare_for_training(df_wrong_type, "text", "date",
                                             "%Y-%m-%d")

    def test_train_word2vec_empty_vocab(self):
        """Test that _train_word2vec() raises RuntimeError when the vocabulary is empty."""
        # Create a DataFrame where each token appears only once; with min_count=5 the vocabulary will be empty.
        df_single = pd.DataFrame({
            "text": [["unique1"], ["unique2"]],
            "date": ["2022-01-01", "2022-01-01"]
            })
        # Create a new instance with a high min_count.
        model_empty = Word2VecSemanticShift(min_count=5, verbose=0)
        with self.assertRaises(RuntimeError):
            # Pass a list of tokenized texts.
            model_empty._train_word2vec(df_single["text"].tolist(),
                                        epochs=1)

    def test_inference_requirements_not_trained(self):
        """Test that calling inference methods before training raises errors."""
        # Without training, both aligned and unaligned get_vector() calls should raise errors.
        with self.assertRaises(ValueError):
            self.model.get_vector("hello", 0, aligned=True)
        with self.assertRaises(ValueError):
            self.model.get_vector("hello", 0, aligned=False)

# -----------------------------------------------------------------------------
# RUN THE TESTS
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()