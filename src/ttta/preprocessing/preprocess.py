from typing import List, Tuple, Union
import numpy as np
from scipy.sparse import hstack, vstack, lil_array, csr_matrix
import itertools
from collections import Counter
import re
from nltk.corpus import stopwords
from scipy.sparse import csr_array, find
from nltk import WordNetLemmatizer
from HanTa import HanoverTagger as ht
from tqdm import tqdm
lemma = WordNetLemmatizer()

"""Implements basic preprocessing functions for text data, such as creating a document-term matrix."""

def preprocess(texts, language="english", individual_stop_word_list=None, verbose=False):
    """Implements a very barebone preprocessing procedure for english texts.

    Is used by the pipeline-method and can be replaced by any arbitrary preprocessing function.

    Args:
        texts: list of texts
        language: language of the texts
        individual_stop_word_list: list of stop words, optional
        verbose: boolean, optional
    Returns:
        list of preprocessed texts
    """
    if not isinstance(language, str):
        raise TypeError("language must be a string!")
    if not isinstance(texts, list):
        try:
            texts = list(texts)
        except ValueError:
            raise TypeError("texts must be a list of strings!")
    if not isinstance(texts[0], str):
        raise TypeError("texts must be a list of strings!")
    if individual_stop_word_list is not None:
        if not isinstance(individual_stop_word_list, list):
            try:
                individual_stop_word_list = list(individual_stop_word_list)
            except ValueError:
                raise TypeError("individual_stop_word_list must be a list of strings!")
        if not isinstance(individual_stop_word_list[0], str):
            raise TypeError("individual_stop_word_list must be a list of strings!")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean!")
    if language == "german":
        tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    elif language == "dutch":
        tagger = ht.HanoverTagger('morphmodel_dutch.pgz')
    elif language == "english":
        tagger = ht.HanoverTagger('morphmodel_en.pgz')
    else:
        tagger = None
    processed_texts = []
    try:
        stop = stopwords.words(language) if individual_stop_word_list is None else individual_stop_word_list
        stop = set([re.sub(r"[^a-zäöüß ]", "", x) for x in stop])
    except:
        stop = set()
    iterator = tqdm(texts) if verbose else texts
    for text in iterator:
        text = re.sub(r"[\s-]+", " ", text)
        text = re.sub(r"[^a-zäöüßA-ZÄÖÜ ]", "", text).split()
        if tagger is not None:
            tagged_text = tagger.tag_sent(text)
            text = [x[1].lower() for x in tagged_text]
        else:
            text = [lemma.lemmatize(x).lower() for x in text]
        text = [x for x in text if x not in stop and len(x) > 2]
        processed_texts.append(text)
    return processed_texts

def create_dtm(texts: List[List[str]], vocab: List[str], min_count: int = 5,
               dtm: np.ndarray = None) -> Tuple[np.ndarray, List[str]]:
    """Create a document-term matrix from a list of texts and updates the existing dtm if there is one.
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
    if dtm is not None:
        if not isinstance(dtm, np.ndarray) and not isinstance(dtm, csr_array):
            raise TypeError("dtm must be a numpy array!")

    counters = [Counter(doc) for doc in texts]
    all_new_words_counted = Counter()
    for obj in counters:
        all_new_words_counted.update(obj)
    # all_new_words_counted = Counter([word for doc in texts for word in doc])
    old_vocab_set = set(vocab)
    new_vocabulary = vocab + list(set([word for word, value in all_new_words_counted.items() if value >= min_count and word not in old_vocab_set]))
    new_vocabulary_index = {word: i for i, word in enumerate(new_vocabulary)}
    new_vocabulary_set = set(new_vocabulary)

    row_indices = []
    col_indices = []
    data = []
    for i, doc in enumerate(texts):
        for word, count in counters[i].items():
            if word in new_vocabulary_set:
                row_indices.append(i)
                col_indices.append(new_vocabulary_index[word])
                data.append(count)
    updated_dtm = csr_array((data, (row_indices, col_indices)),
                             shape=(len(texts), len(new_vocabulary)))

    number_of_new_words = len(new_vocabulary) - len(vocab)
    vocab = new_vocabulary
    if dtm is not None:
        new_data = csr_array((dtm.shape[0], number_of_new_words))
        existing_dtm = hstack((dtm, new_data))
        combined_dtm = vstack((existing_dtm, updated_dtm))
        dtm = combined_dtm#.tocsr()
    else:
        dtm = updated_dtm
    return dtm, vocab

def get_word_and_doc_vector(dtm: Union[csr_array, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a document-term matrix into index vectors.

    The word vector contains the vocabulary index for each word
    occurrence including multiple occurrences in one text. The document vector contains the document index for each
    word occurrence including multiple occurrences in one document.

    Args:
        dtm: document-term matrix
    Returns:
        word_vec: word vector
        doc_vec: document vector
    """
    if not isinstance(dtm, csr_array) and not isinstance(dtm, np.ndarray):
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
