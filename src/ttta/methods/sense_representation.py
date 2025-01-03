"""This module is used to infer and represent different senses of a word across multiple time periods"""
import os.path
import torch
from pathlib import Path
from transformers import BertTokenizer, BertModel
from transformers import logging as lg
from ..preprocessing.schemas import WordSenseEmbedding
from typing import Union, List
import nltk
import logging
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from tqdm import tqdm
from wasabi import msg

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

for corpus in ['wordnet', 'omw-1.4', 'brown']:
    try:
        nltk.find(f'corpora/{corpus}')
    except LookupError:
        logging.info(f'Downloading {corpus} corpus')
        nltk.download(corpus)


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
    model: BertModel
    bert_tokenizer: BertTokenizer
    tokens_tensor: torch.Tensor
    segments_tensors: torch.Tensor
    _tokens: List

    def __init__(
            self,
            pretrained_model_path: Union[str, Path] = None,
    ):
        """Initialize the class with the pretrained model path.

        Args:
            pretrained_model_path: Path to the pretrained model.
        """
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

        lg.set_verbosity_error()
        self._bert_case_preparation()

    @staticmethod
    def load(pretrained_model_path: Union[str, Path] = None) -> 'VectorEmbeddings':
        """
        Loads the pretrained model from the path provided.
        Args:
            pretrained_model_path: Path to the pretrained model.

        Returns:
            VectorEmbeddings
        """
        if pretrained_model_path is not None:
            if not isinstance(pretrained_model_path, str) or not isinstance(pretrained_model_path, Path):
                raise ValueError(
                    f'The path provided is not a valid path: {pretrained_model_path}'
                )
            return VectorEmbeddings(pretrained_model_path)
        return VectorEmbeddings()

    @property
    def tokens(self):
        return self._tokens

    def _bert_case_preparation(self) -> None:
        """This method is used to prepare the BERT model for the inference."""
        model_path = self.model_path if self.model_path is not None else 'bert-base-uncased'
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(
            model_path,
            output_hidden_states=True,
        )
        self.model.eval()
        self.vocab = True

    def infer_vector(self, doc: str, main_word: str) -> torch.Tensor:
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
        if main_word not in doc:
            raise ValueError(
                f'The word: "{main_word}" does not exist in the document: {doc}'
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


class SenseRepresentation:
    """This class is used to recreate the sense representation for a given word
    in a given period of time.

    This implementation is based on the fine-grained sense representations with deep contextualized word embeddings,
    i.e., represent each sense as a distinguished sense embedding. We directly adopt the fine-grained senses
    defined by lexicographers. Comparing with existing diachronic sense studies, our method does not rely on
    human interpretations or mappings to dictionary definitions.

    For a sense \( s_j \) of word \( w_i \),
    we can obtain its example sentences \( \{ \text{Sent}_{wi}^{s_j} \} \) from a dictionary. After feeding them
    into a pre-trained language model, \( w_i \)'s token representations \( \{ e_{wi}^{s_j} \} \) can be retrieved
    from the final hidden layer of the model. The sense embedding \( e_{ij} \) of \( s_j \) is computed by taking
    the average of \( \{ e_{wi}^{s_j} \} \).

    Citation:
    \cite{author2024fine}
    # todo what citation is to be used here?
    """

    target_word: str

    def __init__(
            self,
            target_word: str,
            model_path: Union[str, Path] = None,

    ):
        """Initialize the class with the target word and the model path.

        Args:
            target_word: The main word to extract the sense representation for.
            model_path: The path to the pretrained model.
        """
        self.model = VectorEmbeddings.load(model_path)
        self.target_word = target_word

    def _infer_sentence_embedding(self, sense_examples: List[str]) -> torch.Tensor:
        """Infer the embeddings of the give_word in each example of the sense.

        Args:
            sense_examples: List of examples for the sense.
        Returns: torch.Tensor
        """
        for example in sense_examples:
            yield self.model.infer_vector(
                doc=example,
                main_word=self.target_word
            )

    def _get_senses(self) -> List[str]:
        """Get the senses for the target word.

        Returns: List[str]
        """
        synsets = wn.synsets(self.target_word)
        senses = [synset.definition() for synset in synsets]
        return senses

    def get_sentence_examples(self, sense: int, min_examples=5):
        """Extract or generate at least min_examples sentence examples for a
        specific sense of a word in WordNet.

        Args:
        - sense (int): The sense number of the word.
        - min_examples (int): Minimum number of examples required.

        Returns:
        - List[str]: A list of sentence examples.
        """
        synsets = wn.synsets(self.target_word)

        # Check if the sense number is valid
        if sense <= 0 or sense > len(synsets):
            raise ValueError(f"Sense number {sense} is out of range for the word '{self.target_word}'.")

        # Get the specific synset based on the sense number
        synset = synsets[sense - 1]

        # Retrieve the examples for the synset
        examples = synset.examples()

        # If the number of examples is less than the minimum required, generate more
        if len(examples) < min_examples:
            context_sentences = []
            for sentence in brown.sents():
                if self.target_word in sentence:
                    context_sentences.append(' '.join(sentence))
                if len(context_sentences) >= min_examples:
                    break

            # Add more examples from the Brown corpus if needed
            for context_sentence in context_sentences:
                examples.append(context_sentence)
                if len(examples) >= min_examples:
                    break

        return examples[:min_examples]

    def infer_representation(self) -> List[WordSenseEmbedding]:
        """Infer the mean vector embedding for the given word across all the
        examples in the senses.

        Returns: WordSenseEmbedding object.
        """
        senses = self._get_senses()
        sense_embeddings = []
        for idx, sense in tqdm(enumerate(senses),
                               desc=f'Processing the sense representation for the word: {self.target_word}',
                               total=len(senses)):
            examples = self.get_sentence_examples(idx + 1)
            try:
                msg.good(f'{"-" * 10} Processing the sense: {sense} {"-" * 10}')
                embeddings = list(self._infer_sentence_embedding(examples))
                sense_embedding = torch.mean(torch.stack(embeddings), dim=0).tolist()
                sense_embeddings += [
                    WordSenseEmbedding(
                        id=str(idx),
                        definition=sense,
                        embedding=sense_embedding
                    )
                ]
            except ValueError:
                continue
        return sense_embeddings
