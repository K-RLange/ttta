
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm import tqdm
from transformers import BertForMaskedLM, AdamW, AutoTokenizer
import logging

class NewsExamplesDataset(Dataset):
    """
    Dataset generator for the news examples

    Attributes
    ----------
        examples: List[str]
            List of the examples
        words_to_mask: List[str]
            List of the words to mask

    Methods
    -------
        _pick_masked_embeddings(self, encodings: Dict[str, List[int]]) -> List[int]
            Returns the masked embeddings
        __len__(self) -> int
            Returns the length of the dataset
        __getitem__(self, idx: int) -> Dict[str, torch.Tensor]
            Returns the item at the given index

    """
    def __init__(self,
                 examples:List[str],
                 words_to_mask: List[str],
                 chunk_size=419,

        ):

        print(f'{"-" * 10} Initializing the dataset {"-" * 10}')
        self.examples = examples
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.encodings = self.tokenizer(self.examples, return_tensors='pt', max_length=chunk_size, truncation=True, padding="max_length")
        self.encodings["word_ids"] = [self.encodings.word_ids(i) for i in range(len(self.encodings["input_ids"]))]
        self.encodings["labels"] = self.encodings["input_ids"].detach().clone()
        self.words = self.tokenizer.convert_tokens_to_ids(words_to_mask)

        self.masked_embeddings = self._pick_masked_embeddings(self.encodings['input_ids'])
        self.max_length = chunk_size

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

class BertTrainor():
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
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
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

if __name__ == '__main__':

    with open('../data/all_sentences.txt', 'r') as f:
        sentences = f.read()

    samples = sentences.split('\n')
    data = NewsExamplesDataset(samples, chunk_size=420, words_to_mask=["abuse", "fight", "market"])
    BertTrainor(data, device='cpu', epochs=2).train()

