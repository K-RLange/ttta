import os
import logging
import tqdm
from typing import Union, List
from pathlib import Path
from transformers import AutoTokenizer, BertForMaskedLM, BertTokenizer, BertModel, logging as lg
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np







logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """
    This class is used to prepare the dataset for training.

    Methods
    -------
        __init__(data: List[str], tokenizer, max_length=128, truncation=True, padding="max_length")
            The constructor for the CustomDataset class.
        _pick_masked_embeddings(inputs:torch.Tensor, percentage_size=0.15)
            Returns the list of embeddings that will be masked
        __getitem__(idx)
            Returns the data at the given index
        __len__()
            Returns the length of the dataset
    """

    def __init__(
            self, 
            data: List[str], 
            words_to_mask: List[str],
            max_length=419,
            truncation=True,
            padding="max_length"
            ):


            print(f'{"-" * 10} Initializing the dataset {"-" * 10}')
            self.examples = data
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.encodings = self.tokenizer(self.examples, return_tensors='pt', max_length=max_length, truncation=truncation, padding=padding)
            self.encodings["word_ids"] = [self.encodings.word_ids(i) for i in range(len(self.encodings["input_ids"]))]
            self.encodings["labels"] = self.encodings["input_ids"].detach().clone()
            self.words = self.tokenizer.convert_tokens_to_ids(words_to_mask)

            self.masked_embeddings = self._pick_masked_embeddings(self.encodings['input_ids'])
            self.max_length = max_length

    def _pick_masked_embeddings(self, inputs:torch.Tensor, percentage_size=0.15):
        """
        Returns the list of embeddings that will be masked
        Args:
            inputs: torch.Tensor
                The input tensor
            percentage_size: float
                The percentage of the embeddings to mask
        Returns:
            List[torch.Tensor]
        """
        dim = inputs.shape[0]
        idx = np.random.choice(list(range(dim)), int(dim * percentage_size), replace=False)
        docs = inputs[idx]

        for idx, doc in enumerate(docs):
            doc_ids = doc.numpy().flatten()
            doc_ids[(doc_ids == self.words[0]) | (doc_ids == self.words[1]) | (doc_ids == self.words[2])] = 103
            docs[idx] = torch.tensor(doc_ids)

        return docs
    def __getitem__(self, idx):
        return dict(
            input_ids = self.encodings['input_ids'][idx].clone().detach(),
            attention_mask = self.encodings['attention_mask'][idx].detach().clone(),
            labels = self.encodings['labels'][idx].detach().clone(),
        ).copy()
    def __len__(self):
        return self.encodings.input_ids.shape[0]


class BertTrainer():
    """
    Class for training the Bert model

    Attributes
    ----------
        train_dataset: NewsExamplesDataset
            The dataset to train on
        device: str
            The device to train on
        epochs: int
            The number of epochs to train for

    Methods
    -------
        train(self)
            Trains the model
    """
    def __init__(
            self,
            train_dataset,
            device='cpu',
            epochs=1,
        ):
        self.train_dataset = train_dataset
        self.device = device
        self.epochs = epochs

        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=True,
        )

    def train(self):
        for epoch in range(self.epochs):
            loop = tqdm(self.train_dataloader, leave=True)
            for batch in loop:
                self.optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # process
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        self.model.save_pretrained("bert_model_new")



class WordEmbeddings:
    """
    This class is used to infer the vector embeddings of a word from a sentence.

    Methods
    -------
        infer_vector(doc:str, main_word:str)
            This method is used to infer the vector embeddings of a word from a sentence.
        _bert_case_preparation()
            This method is used to prepare the BERT model for the inference.
    """
    def __init__(
        self,
        pretrained_model_path:Union[str, Path] = None,
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
        """
        This method is used to prepare the BERT model for the inference.
        """
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
        



if __name__ == "__main__":
    pass
