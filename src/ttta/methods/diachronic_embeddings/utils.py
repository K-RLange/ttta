
import json
import os
import time
import logging
from typing import List, Dict
from ttta.preprocessing.semantic import OxfordDictAPI
from ttta.preprocessing.settings import EmbeddingFiles
from ttta.preprocessing.schemas import Words
from pydantic import parse_obj_as


def get_words_and_senses(path:str) -> List[Dict]:
    """
    Get all the words and their senses from the Oxford API, given a path to a file containing the words
    Args:
        path: str
            Path to the file containing the words
    Returns:
        List[Dict]: A list of dictionaries containing the words and their senses

    """
    all_words = []
    if not os.path.exists(path):
        raise ValueError(
            f'No matches for the given path: {path}'
        )
    with open(path) as f: full_text = f.read()
    for idx, word in enumerate(full_text.split('\n')):
        try:
            all_words += [OxfordDictAPI(word_id=word).get_senses()]
            if (idx != 0) and (idx % 10 == 0):
                logging.info('Waiting for 3 seconds before continuing...')
                time.sleep(3)
        except ValueError:
            continue

    if len(all_words) != len(full_text.split('\n')):
        logging.warning(
            f'Number of words processed: {len(all_words)} is inferior to the number of words in the file: {path}')

    return all_words


def poly_words(oxford_words_file: str):
    """
    Get all words that have more than one sense
    Args:
        oxford_words:

    Returns:
        str: the words that have more than one sense
    """
    with open(oxford_words_file) as f: oxford_words = parse_obj_as(List[Words], json.load(f))
    for word in oxford_words:
        if len(word.senses) > 1:
            yield word.word



if __name__ == '__main__':
    files = EmbeddingFiles()

    with open(files.oxford_word_senses, 'w') as f:
        p = get_words_and_senses(files.poly_words_f)
        json.dump(p, f, indent=4)

