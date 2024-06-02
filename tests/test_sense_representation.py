import unittest
from unittest.mock import patch, MagicMock
import torch
from pathlib import Path
from src.ttta.preprocessing.schemas import WordSenseEmbedding
from src.ttta.methods.sense_representation import VectorEmbeddings, SenseRepresentation


# Assuming the classes VectorEmbeddings and SenseRepresentation are defined as shown above

class TestVectorEmbeddings(unittest.TestCase):

    @patch('transformers.BertModel.from_pretrained')
    @patch('transformers.BertTokenizer.from_pretrained')
    def setUp(self, mock_tokenizer, mock_model):
        self.mock_tokenizer = mock_tokenizer
        self.mock_model = mock_model
        self.mock_tokenizer.return_value = MagicMock()
        self.mock_model.return_value = MagicMock()
        self.model_path = "bert-base-uncased"
        self.vector_embeddings = VectorEmbeddings(self.model_path)

    def test_initialization(self):
        self.assertIsInstance(self.vector_embeddings, VectorEmbeddings)
        self.assertEqual(self.vector_embeddings.model_path, Path(self.model_path))
        self.assertTrue(self.vector_embeddings.vocab)

    def test_infer_vector(self):
        doc = "The bank will not provide the loan."
        main_word = "bank"

        self.vector_embeddings.bert_tokenizer.tokenize.return_value = ["[CLS]", "the", "bank", "will", "not", "provide",
                                                                       "the", "loan", ".", "[SEP]"]
        self.vector_embeddings.bert_tokenizer.convert_tokens_to_ids.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        output_tensor = torch.rand((1, 10, 768))
        self.vector_embeddings.model.return_value = (None, None, [output_tensor])

        inferred_vector = self.vector_embeddings.infer_vector(doc, main_word)
        self.assertIsInstance(inferred_vector, torch.Tensor)

    def test_load(self):
        vector_embeddings = VectorEmbeddings.load(self.model_path)
        self.assertIsInstance(vector_embeddings, VectorEmbeddings)
        self.assertEqual(vector_embeddings.model_path, Path(self.model_path))


class TestSenseRepresentation(unittest.TestCase):

    @patch('transformers.BertModel.from_pretrained')
    @patch('transformers.BertTokenizer.from_pretrained')
    def setUp(self, mock_tokenizer, mock_model):
        self.mock_tokenizer = mock_tokenizer
        self.mock_model = mock_model
        self.mock_tokenizer.return_value = MagicMock()
        self.mock_model.return_value = MagicMock()
        self.target_word = "bank"
        self.model_path = "bert-base-uncased"
        self.sense_representation = SenseRepresentation(self.target_word, self.model_path)

    def test_initialization(self):
        self.assertIsInstance(self.sense_representation, SenseRepresentation)
        self.assertEqual(self.sense_representation.target_word, self.target_word)
        self.assertTrue(self.sense_representation.model.vocab)

    @patch('nltk.corpus.wordnet.synsets')
    def test_get_senses(self, mock_synsets):
        mock_synset = MagicMock()
        mock_synset.definition.return_value = "A financial institution that accepts deposits and channels the money into lending activities."
        mock_synsets.return_value = [mock_synset]

        senses = self.sense_representation._get_senses()
        self.assertEqual(len(senses), 1)
        self.assertEqual(senses[0], mock_synset.definition())

    @patch('nltk.corpus.wordnet.synsets')
    @patch('nltk.corpus.brown.sents')
    def test_get_sentence_examples(self, mock_brown_sents, mock_synsets):
        mock_synset = MagicMock()
        mock_synset.examples.return_value = ["The bank will not provide the loan."]
        mock_synsets.return_value = [mock_synset]
        mock_brown_sents.return_value = [["The", "bank", "is", "near", "the", "river"]]

        examples = self.sense_representation.get_sentence_examples(1, min_examples=5)
        self.assertGreaterEqual(len(examples), 5)

    @patch('torch.mean')
    @patch('SenseRepresentation._infer_sentence_embedding')
    @patch('SenseRepresentation._get_senses')
    def test_infer_representation(self, mock_get_senses, mock_infer_sentence_embedding, mock_mean):
        mock_get_senses.return_value = [
            "A financial institution that accepts deposits and channels the money into lending activities."]
        mock_infer_sentence_embedding.return_value = [torch.rand((768,))]
        mock_mean.return_value = torch.rand((768,))

        embeddings = self.sense_representation.infer_representation()
        self.assertIsInstance(embeddings, list)
        self.assertGreater(len(embeddings), 0)
        self.assertIsInstance(embeddings[0], WordSenseEmbedding)


if __name__ == '__main__':
    unittest.main()
