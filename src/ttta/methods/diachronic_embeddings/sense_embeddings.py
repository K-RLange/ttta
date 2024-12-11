import os.path

import torch
from pathlib import Path
from transformers import BertTokenizer, BertModel
from transformers import logging as lg
from ttta.preprocessing.schemas import OxfordAPIResponse, WordSenseEmbedding
from ttta.preprocessing.settings import FileLoader
import logging
from typing import Union


class VectorEmbeddings:
    """This class is used to infer the vector embeddings of a word from a
    sentence.

    Methods
    -------
        infer_vector(doc:str, main_word:str)
            This method is used to infer the vector embeddings of a word from a sentence.
        _bert_case_preparation()
            This method is used to prepare the BERT model for the inference.
    """
    def __init__(
        self,
        pretrained_model_path: Union[str, Path] = None,
    ):
        self.model_path = pretrained_model_path
        if pretrained_model_path is not None:
            if not os.path.exists(pretrained_model_path):
                raise ValueError(
                    f'The path {pretrained_model_path} does not exist'
                )
            self.model_path = Path(pretrained_model_path)

        self._tokens = []
        self.model = None
        self.vocab = False
        self.lematizer = None

        lg.set_verbosity_error()
        self._bert_case_preparation()

    @property
    def tokens(self):
        return self._tokens

    def _bert_case_preparation(self) -> None:
        """This method is used to prepare the BERT model for the inference."""
        model_path = self.model_path if self.model_path is not None else 'bert-base-uncased'
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(
            model_path,
            output_hidden_states = True,
        )
        self.model.eval()
        self.vocab = True

    def infer_vector(self, doc:str, main_word:str):
        """
        This method is used to infer the vector embeddings of a word from a sentence.
        Args:
            doc: Document to process
            main_word: Main work to extract the vector embeddings for.

        Returns: torch.Tensor

        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        marked_text = "[CLS] " + doc + " [SEP]"
        tokens = self.bert_tokenizer.tokenize(marked_text)
        try:
            main_token_id = tokens.index(main_word.lower())
            idx = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            segment_id = [1] * len(tokens)

            self.tokens_tensor = torch.tensor([idx])
            self.segments_tensors = torch.tensor([segment_id])

            with torch.no_grad():
                outputs = self.model(self.tokens_tensor, self.segments_tensors)
                hidden_states = outputs[2]

            return hidden_states[-2][0][main_token_id]

        except ValueError:
            raise ValueError(
                f'The word: "{main_word}" does not exist in the list of tokens: {tokens} from {doc}'
            )


class ExtractSenseEmbeddings:
    """Wrapper class for the Vector embeddings that is used to extract the
    embeddings for all the senses.

    Attributes
    ----------
        sense: dict
            The sense for which the embeddings are to be extracted.
        word: str
            The word for which the embeddings are to be extracted.
        vector_embeddings: VectorEmbeddings
            The class that is used to extract the embeddings.
        api_component: OxfordAPIResponse
            The class that is used to extract the senses.
        all_words: list
            The list of all the words for which the embeddings are to be extracted.

    Methods
    -------
        __call__(sense:dict, main_w:str)
            This method is used to initialize the object.
        _infer_sentence_vector()
            This method is used to infer the vector embeddings for all the sentences in the sense.
        infer_mean_vector()
            This method is used to infer the mean vector embeddings for all the sentences in the sense.
        create_sense_embeddings()
            This method is used to extract the embeddings for all the senses.
    """
    def __init__(
            self
        ):
        self.sense = None
        self.word = None
        self.vector_embeddings = VectorEmbeddings()
        self.api_component = OxfordAPIResponse()
        self.all_words = FileLoader.load_files(self.__class__.__name__)

    def __call__(self, sense:dict, given_word:str) -> 'ExtractSenseEmbeddings':
        """
        This method is used to initialize the object with the particular senses and the given word.
        Args:
            sense: The sense for which the embeddings are to be extracted.
            given_word: The word to track in each examples for all senses of the word.

        Returns: The self object initialized.

        """
        if not isinstance(sense, OxfordAPIResponse):
            raise ValueError(
                f'Expected type {OxfordAPIResponse.__class__} for the sense, but got type: {type(sense)}'
            )
        self.sense = sense
        self.word = given_word
        return self

    def _infer_sentence_embedding(self) -> torch.Tensor:
        """Infer the embeddings of the give_word in each example of the sense.

        Returns: torch.Tensor
        """
        for example in self.sense.examples:
            yield self.vector_embeddings.infer_vector(
                doc=example,
                main_word=self.word
            )

    def infer_mean_vector(self) -> WordSenseEmbedding:
        """Infer the mean vector embedding for the given word across all the
        examples in the senses.

        Returns: WordSenseEmbedding object.
        """
        all_token_embeddings =  torch.stack(list(self._infer_sentence_embedding()))
        return WordSenseEmbedding(
            id=self.sense.id,
            definition=self.sense.definition,
            embedding=torch.mean(all_token_embeddings, dim=0).tolist(),
        )

    def create_sense_embeddings(self) -> list:
        """Extract the averaged vector embedding for a list of polysemous words
        across a wide range of contexts.

        Returns:
            list: A list of dictionaries containing the word and the sense embeddings.
        """
        all_embeddings = []
        word_embedding = {}
        for word_obj in self.all_words:
            logging.info(f'{"-"*40} Embedding the word {word_obj.word} {"-"*40} ')
            word_embedding['word'] = word_obj.word
            word_embedding['senses'] = [self(sens, word_obj.word).infer_mean_vector() for sens in word_obj.senses]
            all_embeddings += [word_embedding.copy()]

        return all_embeddings