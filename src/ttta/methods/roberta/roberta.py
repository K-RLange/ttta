import os
import logging
import tqdm
from typing import Union, List, Optional
from pathlib import Path
import math
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup, logging as lg
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from accelerate import Accelerator
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import collections
import random
# import nltk
# nltk.download('averaged_perceptron_tagger')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """This class is used to create a custom dataset for the Roberta model. It
    inherits from torch.utils.data.Dataset.

    Methods
    -------
        __init__(data: List[str], tokenizer, max_length=128, truncation=True, padding=True)
            The constructor for the CustomDataset class.
        __len__()
            This method is used to get the length of the dataset.
        __getitem__(idx)
            This method is used to get the item at a specific index.
    """
    def __init__(
            self, 
            data: List[str], 
            tokenizer, 
            max_length=128,
            truncation=True,
            padding= "max_length",
            ):
        """
        Args:
            data (List[str]): List of strings to create a dataset from.
            tokenizer: Tokenizer to tokenize the data with.
            max_length (int): Maximum length of the input sequence. Defaults to 128.
            truncation (bool): Whether to truncate the input sequence to max_length or not. Defaults to True.
            padding (str): Whether to pad the input sequence to max_length or not. Defaults to "max_length".
        
        Attributes:
            tokenizer: Tokenizer to tokenize the data with.
            max_length (int): Maximum length of the input sequence. Defaults to 128.
            tokenized_data (dict): Dictionary containing the input_ids, attention_mask, and labels. 
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_data = tokenizer(data, truncation=truncation, padding=padding, max_length=max_length)

    def __len__(self):
        return len(self.tokenized_data.input_ids)

    def __getitem__(self, idx):
        """Retrieves the item at the specified index.

        Parameters:
            idx (int): Index of the item to retrieve.

        Returns:
            tokenized_data (dict): Dictionary containing the input_ids, attention_mask, and labels.
        """
        # Get the tokenized inputs at the specified index
        input_ids = self.tokenized_data.input_ids[idx]
        attention_mask = self.tokenized_data.attention_mask[idx]

        # Return a dictionary containing input_ids, attention_mask, and labels (if applicable)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
            # Add 'labels': labels if you have labels for your data
        }

class RobertaTrainer:
    """This class is used to train a Roberta model.

    Methods
    -------
        __init__(model_name="roberta-base", max_length=128, mlm_probability=0.15, batch_size=4, learning_rate=1e-5, epochs=3, warmup_steps=500, split_ratio=0.8)
            The constructor for the RobertaTrainer class.
        prepare_dataset(data)
            This method is used to prepare the dataset for training.
        train(data, output_dir: Union[str, Path] = None)
            This method is used to train the model.
    """
    def __init__(
            self, 
            model_name: str = "roberta-base", 
            max_length: int = 128, 
            mlm_probability: float = 0.15, 
            batch_size: int = 4, 
            learning_rate: float = 1e-5, 
            epochs: int = 3, 
            warmup_steps: int = 500, 
            split_ratio: float = 0.8, 
            truncation: bool = True, 
            padding: str = "max_length"
            ):

        """
        Args:
            model_name (str): Name of the model to train. Defaults to "roberta-base".
            max_length (int): Maximum length of the input sequence. Defaults to 128.
            mlm_probability (float): Probability of masking tokens in the input sequence. Defaults to 0.15.
            batch_size (int): Size of the batch. Defaults to 4.
            learning_rate (float): Learning rate of the optimizer. Defaults to 1e-5.
            epochs (int): Number of epochs to train the model for. Defaults to 3.
            warmup_steps (int): Number of warmup steps for the learning rate scheduler. Defaults to 500.
            split_ratio (float): Ratio to split the data into train and test. Defaults to 0.8.
            truncation (bool): Whether to truncate the input sequence to max_length or not. Defaults to True.
            padding (str): Whether to pad the input sequence to max_length or not. Defaults to "max_length".
        
        Attributes:
            tokenizer (transformers.RobertaTokenizer): Tokenizer to tokenize the data with.
            model (transformers.RobertaForMaskedLM): Model to train.
            data_collator (transformers.DataCollatorForLanguageModeling): DataCollatorForLanguageModeling object to collate the data.
            split_ratio (float): Ratio to split the data into train and test. Defaults to 0.8.
            truncation (bool): Whether to truncate the input sequence to max_length or not. Defaults to True.
            padding (str): Whether to pad the input sequence to max_length or not. Defaults to "max_length".
            max_length (int): Maximum length of the input sequence. Defaults to 128.
            batch_size (int): Size of the batch. Defaults to 4.
            learning_rate (float): Learning rate of the optimizer. Defaults to 1e-5.
            epochs (int): Number of epochs to train the model for. Defaults to 3.
            warmup_steps (int): Number of warmup steps for the learning rate scheduler. Defaults to 500.
            accelerator (accelerate.Accelerator): Accelerator object to distribute the training across multiple GPUs.
        """
        
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=mlm_probability
            )

        self.split_ratio = split_ratio
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.accelerator = Accelerator()

    def prepare_dataset(self, data: List[str]):
        """
        This method is used to prepare the dataset for training.
        Args:
            data: List of strings to train the model on.
            
        Returns:
            train_loader (torch.utils.data.DataLoader): DataLoader object containing the training data.
            dataset (CustomDataset): CustomDataset object containing the training data.
        """
        dataset = CustomDataset(
            data, 
            self.tokenizer, 
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding
            )
        
        train_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.data_collator
            )
        
        return train_loader, dataset

    def train(
            self, 
            data: List[str],
            output_dir: Optional[Union[str, Path]] = None
            ) -> None:
        """This method is used to train the model.

        Args:
            data (List[str]): List of strings to train the model on.
            output_dir (str, Path, None): Path to save the model to. Defaults to None.
        

        Examples:
            >>> model = RobertaTrainer(epoch=3)
            >>> model.train(data=["The brown fox jumps over the lazy dog", "The brown fox jumps over the lazy dog", "Hello world!"], output_dir="../../output/MLM_roberta")
            Epoch: 0 | Loss: 1.1637206077575684 | Perplexity: 3.2020153999328613
            Epoch: 1 | Loss: 0.6941609382629395 | Perplexity: 2.0011680126190186
            Epoch: 2 | Loss: 0.4749067425727844 | Perplexity: 1.608262062072754
        """
        
        train_data, test_data = train_test_split(
            data, 
            test_ratio= 1 - self.split_ratio, 
            random_seed=42
            )
        
        train_loader, _ = self.prepare_dataset(train_data)
        test_loader, _ = self.prepare_dataset(test_data)
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate
            )

        model, optimizer, train_loader, test_loader = self.accelerator.prepare(
            self.model, 
            optimizer, 
            train_loader, 
            test_loader
            )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.warmup_steps, 
            num_training_steps=len(train_loader) * self.epochs
            )

        progress_bar = tqdm.tqdm(
            range(len(train_loader) * self.epochs), 
            desc="Training", 
            dynamic_ncols=True
            )
        
        for epoch in range(self.epochs):
            self.model.train()

            for batch in train_loader:
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                optimizer.step()
                scheduler.step()  # Update learning rate scheduler
                optimizer.zero_grad()
                progress_bar.update(1)

            self.model.eval()
            losses = []
            for step, batch in enumerate(test_loader):
                with torch.no_grad():
                    outputs = self.model(**batch)
                
                loss = outputs.loss
                losses.append(self.accelerator.gather(loss.repeat(self.batch_size)))
            
            losses = torch.cat(losses)
            losses = losses[:len(test_data)]

            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")
            print(f"Epoch: {epoch} | Loss: {torch.mean(losses)} | Perplexity: {perplexity}")

            # Save model
            if output_dir is not None:
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=self.accelerator.save)
                if self.accelerator.is_main_process:
                    self.tokenizer.save_pretrained(output_dir)






class RobertaEmbedding:
    """This class is used to infer vector embeddings from a document.

    Methods
    -------
        __init__(pretrained_model_path:Union[str, Path] = None)
            The constructor for the VectorEmbeddings class.
        _roberta_case_preparation()
            This method is used to prepare the Roberta model for the inference.
        infer_vector(doc:str, main_word:str)
            This method is used to infer the vector embeddings of a word from a document.
        infer_mask_logits(doc:str)
            This method is used to infer the logits of a word from a document.
    """
    def __init__(
        self,
        pretrained_model_path: Union[str, Path],
    ):
        """
        Args:
            pretrained_model_path (str, Path, None): Path to the pretrained model. Defaults to None.

        Attributes:
            model_path (str, Path, None): Path to the pretrained model. Defaults to None.
            model (transformers.RobertaModel): RobertaModel object to infer vector embeddings from.
            MLM (transformers.RobertaForMaskedLM): RobertaForMaskedLM object to infer vector embeddings from.
            tokenizer (transformers.RobertaTokenizer): Tokenizer to tokenize the data with.
            max_length (int): Maximum length of the input sequence. Defaults to 128.
            vocab (bool): Whether the model has been initialized or not.
        """
        self.model_path = pretrained_model_path
        self._tokens = []
        self.model = None
        self.vocab = False

        lg.set_verbosity_error()
        self._roberta_case_preparation()

    @property
    def tokens(self):
        return self._tokens

    def _roberta_case_preparation(self) -> None:
        """This method is used to prepare the BERT model for the inference."""
        model_path = self.model_path if os.path.exists(self.model_path) else 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaModel.from_pretrained(
            model_path, 
            output_hidden_states=True
            )
        self.MLM = RobertaForMaskedLM.from_pretrained(
            model_path
        )
        # self.model_max_length = self.model.config.max_position_embeddings
        # self.mlm_max_length = self.MLM.config.max_position_embeddings
        self.model.eval()
        self.MLM.eval()
        self.vocab = True

    def infer_vector(self, doc:str, main_word:str) -> Optional[torch.Tensor]:
        """This method is used to infer the vector embeddings of a word from a
        document.

        Args:
            doc (str): Document to process
            main_word (str): Main work to extract the vector embeddings for.

        Returns: 
            embeddings (torch.Tensor): Tensor of stacked embeddings of shape (num_embeddings, embedding_size) where num_embeddings is the number of times the main_word appears in the doc.

        Examples:
            >>> model = RobertaEmbedding()
            >>> model.infer_vector(doc="The brown fox jumps over the lazy dog", main_word="fox")
            tensor([[-0.2182, ..., -0.1709],
                    ...,
                    [-0.2182, ..., -0.1706]])
        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        
     
        input_ids = self.tokenizer(doc, return_tensors="pt", max_length=512, truncation=True).input_ids
        token = self.tokenizer.encode(main_word, add_special_tokens=False)[0]
        word_token_index = torch.where(input_ids == token)[1]

        try:
            with torch.no_grad():
                embeddings = self.model(input_ids).last_hidden_state
               
            emb = [embeddings[0, idx] for idx in word_token_index]
            return torch.stack(emb).mean(dim=0)
        
        except IndexError:
            raise ValueError(f'The word: "{main_word}" does not exist in the list of tokens')



    
    def infer_mask_logits(self, doc:str) -> Optional[torch.Tensor]:
        """This method is used to infer the logits of the mask token in a
        document.

        Args:
            doc (str): Document to process where the mask token is present.

        Returns: 
            logits (Optional[torch.Tensor]): Tensor of stacked logits of shape (num_embeddings, logits_size) where num_embeddings is the number of times the mask token appears in the doc withing the max_length.

        Examples:
            >>> model = RobertaEmbedding()
            >>> model.infer_mask_logits(doc="The brown fox <mask> over the lazy dog")
            tensor([[-2.1816e-01,  ..., -1.7064e-01],
                    ...,
                    [-2.1816e-01, ..., -1.7093e-01]])
        """

        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.MLM.__class__.__name__} has not been initialized'
            )

        input_ids = self.tokenizer(doc, return_tensors="pt", max_length= 512, truncation=True).input_ids
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
        l = []
        try:
            with torch.no_grad():
                logits = self.MLM(input_ids).logits
                
            l = [logits[0, idx] for idx in mask_token_index]
            return torch.stack(l) if len(l) > 0 else torch.empty(0)
        
        except IndexError:
            raise ValueError(f'The mask falls outside of the max length of {512}, please use a smaller document')

        




class RobertaInference:
    """Wrapper class for the RobertaEmbedding class for inference.

    Methods
    -------
        __init__(pretrained_model_path:Union[str, Path] = None)
            The constructor for the VectorEmbeddings class.
        _roberta_case_preparation()
            This method is used to prepare the Roberta model for the inference.
        get_embedding(word:str, doc: Optional[Union[str, List[str]]] = None,, mask:bool=False)
            This method is used to infer the vector embeddings of a word from a document.
        get_top_k_words(word:str, doc:str, k:int=3)
            This method is used to infer the vector embeddings of a word from a document.
    """

    def __init__(
            self,
            pretrained_model_path: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            pretrained_model_path (str, Path, None): Path to the pretrained model. Defaults to None.
        
        Attributes:
            model_path (str, Path, None): Path to the pretrained model. Defaults to None.
            word_vectorizor (RobertaEmbedding): RobertaEmbedding object to infer vector embeddings from.
            vocab (bool): Whether the model has been initialized or not.
        """
        self.model_path = pretrained_model_path
        if pretrained_model_path is not None:
            if not os.path.exists(pretrained_model_path):
                raise ValueError(
                    f'The path {pretrained_model_path} does not exist'
                )
            self.model_path = Path(pretrained_model_path)

        self.word_vectorizor = None
        self.vocab = False
        

        lg.set_verbosity_error()
        self._roberta_case_preparation()
    
    
    def _roberta_case_preparation(self) -> None:
        """This method is used to prepare the Roberta model for the
        inference."""
        model_path = self.model_path if self.model_path is not None else 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.word_vectorizor = RobertaEmbedding(pretrained_model_path=model_path)
        self.vocab = True

    
    def get_embedding(
            self,
            main_word : str, 
            doc: Optional[str] = None,
            mask : bool = False
            ) -> torch.Tensor:
        
        """This method is used to infer the vector embeddings of a word from a
        document.

        Args:
            main_word (str): Word to get the vector embeddings for
            doc (str, None): Documents to get the vector embeddings of the main_word from. If None, the document is the main_word itself. Defaults to None.
            mask: Whether to mask the main_word in the documents or not. Defaults to False.
            
        Returns: 
            embeddings (torch.Tensor): Tensor of stacked embeddings of shape (num_embeddings, embedding_size) where num_embeddings is the number of times the main_word appears in the doc, depending on the mask parameter.

        Examples:
            >>> model = RobertaInference()
            >>> model.get_embedding(main_word="office", doc="The brown office is very big", mask=False)
            tensor([[-0.2182, ..., -0.1709],
                    ...,
                    [-0.2182, ..., -0.1706]])
        """
        
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.word_vectorizor.__class__.__name__} has not been initialized'
            )
        
        doc = self._cutDoc(doc=doc, main_word=main_word)
        if doc is None:
            emb = torch.empty(0)
        
        
        if mask:
            doc = doc.replace(main_word, self.tokenizer.mask_token)
            main_word = self.tokenizer.mask_token
        
        else:
            main_word = ' ' + main_word.strip()
            
        try:
            emb = self.word_vectorizor.infer_vector(doc=doc, main_word=main_word)
            return emb
        
        except ValueError:
            # print(f'The word: "{main_word}" does not exist in the list of tokens')
            return torch.empty(0)

    def get_top_k_words(
            self,
            main_word : str,
            doc: str,
            k: int = 3,
            pot_tag: Union[bool, str, List[str]] = False
            ) -> List[str]:
        """
        This method is used to infer the vector embeddings of a main_word from a document.
        Args:
            main_word: Word to mask
            doc: Document to infer the top k words of the main_word from
            k: Number of top words to return

        Returns:
            top_k_words (List[str]): List of top k words

        Examples:
            >>> model = RobertaInference()
            >>> model.get_top_k_words(main_word="office", doc="The brown office is very big")
            ['room', 'eye', 'bear']
        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.word_vectorizor.__class__.__name__} has not been initialized'
            )
        
        doc = self._cutDoc(doc=doc, main_word=main_word)
        if doc is None:
            return []

        masked_doc = doc.replace(main_word, '<mask>')
        try:
            logits = self.word_vectorizor.infer_mask_logits(doc=masked_doc)
            top_k = []

            for logit_set in logits:
                top_k_tokens = torch.topk(logit_set, k).indices
                top_k_words = [self.tokenizer.decode(token.item()).strip() for token in top_k_tokens]
                if isinstance(pot_tag, str):
                    pot_tag = [pot_tag]
                
                if isinstance(pot_tag, list):
                    top_k_words = [word for word in top_k_words if pos_tag([word])[0][1] in pot_tag]
                    
                elif pot_tag:
                    main_pos = pos_tag([main_word])[0][1]
                    top_k_words = [word for word in top_k_words if pos_tag([word])[0][1] == main_pos]

                top_k.extend(top_k_words)

            # rank the top_k words by frequency
            top_k = [word for word in top_k if word not in stopwords.words('english')]
            top_k = collections.Counter(top_k)
            top_k = [word for word, _ in top_k.most_common(k)]
            return top_k
        
        except ValueError:
            print(f'The word: "{main_word}" does not exist in the list of tokens')
            return []


    def _cutDoc(
            self, 
            main_word: str, 
            doc: Optional[str] = None, 
            max_length: int = 512
            ) -> Optional[str]:

        main_word = ' ' + main_word.strip() + ' '
        if doc is None:
            doc = main_word
            return doc
        
        tokens = self.tokenizer.tokenize(doc)
        main_token = self.tokenizer.tokenize(main_word)[0]
       
        try:
            main_index = tokens.index(main_token)
        
        except ValueError:
            return None
        
        start = max(0, main_index - max_length//2)
        end = start + max_length

        if end > len(tokens):
            end = len(tokens)
            start = max(0, end - max_length)

        # Convert the tokens back to ids for decoding
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens[start:end])

        # Decode the token ids back to a string
        sliced_doc = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return sliced_doc


def train_test_split(data: List[str], test_ratio=0.2, random_seed=None):
    """Split the data into train and test sets.

    Args:
        data (List[str]): The data to split.
        test_ratio (float): The ratio of the test set.
        random_seed (int): The random seed.

    Returns:
        data Tuple[List[str], List[str]]: The training and testing datasets.
    """
    
    if random_seed:
        random.seed(random_seed)
    data_copy = data[:]
    random.shuffle(data_copy)
    split_idx = int(len(data_copy) * (1 - test_ratio))
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]
    return train_data, test_data
