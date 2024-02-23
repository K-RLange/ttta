import os
from glob import glob
from typing import List, Tuple, Union
import numpy as np
from scipy.sparse import lil_matrix, hstack, vstack
import itertools
from collections import Counter
from .utils.utils import *
import re
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, find
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
stop = stopwords.words("english")  # We use english texts
stop = [re.sub(r"[^a-z ]", "", x) for x in stop]


class PREPROCESS():

    def __init__(self, config_path='pre.yaml'):
        
        self.configs = read_yaml(config_path)
        self.path = self.configs['input_path']

        if str(self.configs['periods']).lower() == 'all':
            self.file_paths = glob(os.path.join(self.path, '*.xml'))
            self.time_periods = sorted([i.split('/')[-1].split('.')[1] for i in self.file_paths])
        
        else:
            self.time_periods = self.configs['periods']
            self.file_paths = [i for i in glob(os.path.join(self.path, '*.xml')) if i.split('/')[-1].split('.')[1] in self.time_periods]


        self.periods = len(self.time_periods)
        print(f'Found {self.periods} periods')

        for i, p in enumerate(self.time_periods):
            print(f'Processing {p} period')
            articles = get_articles(self.file_paths[i])
            # sentences = get_sentences(articles)

            print(f'Found {len(articles)} articles')

            
            if not self.configs['Preprocessing']['skip']:
                print('Preprocessing articles')
                clean_articles = [cleantxt(a, self.configs['Preprocessing']['options']) for a in articles]

            else:
                clean_articles = articles
            

            if self.configs['Preprocessing']['save_as'] == 'articles':
                print(f'Saving {len(clean_articles)} articles at {self.configs["Preprocessing"]["output_path"]}/{p}_articles.txt')
                save_texts(clean_articles, f'{self.configs["Preprocessing"]["output_path"]}/{p}_articles.txt')
            
            print(f'Finished processing {p} period\n\n\n')




def preprocess(texts):
    """
    Implements a very barebone preprocessing procedure for english texts. Is used by the pipeline-method and can be replaced by any arbitrary preprocessing function.
    Args:
        texts: list of texts
    Returns:
        list of preprocessed texts
    """
    if not isinstance(texts, list):
        try:
            texts = list(texts)
        except ValueError:
            raise TypeError("texts must be a list of strings!")
    if not isinstance(texts[0], str):
        try:
            texts = [str(x) for x in texts]
        except ValueError:
            raise TypeError("texts must be a list of strings!")
    processed_texts = []
    for text in texts:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z ]", "", text).split()
        text = [x for x in text if x not in stop and len(x) > 2]
        text = [lemma.lemmatize(x) for x in text]
        processed_texts.append(text)
    return processed_texts

def create_dtm(texts: List[List[str]], vocab: List[str], min_count: int = 5, deleted_indices=None, dtm: np.ndarray = None) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Creates a document-term matrix from a list of texts and updates the existing dtm if there is one.
    Stores both the dtm and the vocabulary in the class.
    Args:
        texts: list of texts
    Returns:
        None
    """
    if not isinstance(texts, list):
        try:
            texts = list(texts)
        except ValueError:
            raise TypeError("texts must be a list of list of strings!")
    if not isinstance(texts[0], list):
        try:
            texts = [list(x) for x in texts]
        except ValueError:
            raise TypeError("texts must be a list of lists of strings!")
    if not isinstance(texts[0][0], str):
        try:
            texts = [[str(x) for x in text] for text in texts]
        except ValueError:
            raise TypeError("texts must be a list of lists of strings!")
    if not isinstance(vocab, list):
        try:
            vocab = list(vocab)
        except ValueError:
            raise TypeError("vocab must be a set of strings!")
    if not isinstance(min_count, int):
        try:
            min_count = int(min_count)
        except ValueError:
            raise TypeError("min_count must be an integer!")
    if deleted_indices is None:
        deleted_indices = []
    if not isinstance(deleted_indices, list):
        raise TypeError("deleted_indices must be a list of integers!")
    if not isinstance(deleted_indices[0], int):
        raise TypeError("deleted_indices must be a list of integers!")
    if dtm is not None:
        if not isinstance(dtm, np.ndarray):
            raise TypeError("dtm must be a numpy array!")

    all_new_words_counted = Counter([word for doc in texts for word in doc])
    old_vocab_set = set(vocab)
    new_vocabulary = vocab + list(set([word for word, value in all_new_words_counted.items() if value >= min_count and word not in old_vocab_set]))
    new_vocabulary_index = {word: i for i, word in enumerate(new_vocabulary)}
    new_vocabulary_set = set(new_vocabulary)

    non_deleted_bool = [len([word for word in text if word in new_vocabulary_set]) > 0 for i, text in enumerate(texts)]
    non_deleted_indices = np.argwhere(non_deleted_bool).flatten()
    deleted_indices.extend(non_deleted_indices + max(deleted_indices + [0]))
    texts = list(itertools.compress(texts, non_deleted_bool))

    updated_dtm = lil_matrix((len(texts), len(new_vocabulary)))
    for i, doc in enumerate(texts):
        word_counts = Counter(doc)
        updated_dtm[i, [new_vocabulary_index[word] for word in word_counts.keys() if word in new_vocabulary_set]] = \
            [value for word, value in word_counts.items() if word in new_vocabulary_set]

    number_of_new_words = len(new_vocabulary) - len(vocab)
    vocab = new_vocabulary
    if dtm is not None:
        existing_dtm = hstack((dtm, np.zeros((dtm.shape[0], number_of_new_words))))
        combined_dtm = vstack((existing_dtm, updated_dtm))
        dtm = combined_dtm.tocsr()
    else:
        dtm = updated_dtm.tocsr()
    return dtm, vocab, deleted_indices

def get_word_and_doc_vector(dtm: Union[csr_matrix, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turns a document-term matrix into index vectors. The word vector contains the vocabulary index for each word
    occurrence including multiple occurrences in one text. The document vector contains the document index for each
    word occurrence including multiple occurrences in one document.
    Args:
        dtm: document-term matrix
    Returns:
        word_vec: word vector
        doc_vec: document vector
    """
    if not isinstance(dtm, csr_matrix) or isinstance(dtm, np.ndarray):
        try:
            dtm = np.array(dtm)
            if len(dtm.shape) == 0:
                raise ValueError
        except ValueError:
            raise TypeError("dtm must be a numpy array!")
    if dtm.shape[0] == 0 or dtm.shape[1] == 0:
        raise ValueError("dtm must not be empty!")
    (row_nonzero, column_nonzero) = dtm.nonzero()
    _, _, value_list = find(dtm)
    value_list = value_list.astype(int)

    word_vec = []
    doc_vec = []
    for i, elem in enumerate(row_nonzero):
        word_vec += [column_nonzero[i]] * value_list[i]
        doc_vec += [elem] * value_list[i]
    return np.array(word_vec, dtype=np.uint64), np.array(doc_vec, dtype=np.uint64)


if __name__ == '__main__':
    preprocess = PREPROCESS()
    print(*preprocess.time_periods)