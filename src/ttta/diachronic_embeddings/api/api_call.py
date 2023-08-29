import requests
from requests.models import Response
import json
from typing import Dict
from itertools import chain
import re
from nltk import WordNetLemmatizer
from src.settings  import OxfordAPISettings
import logging
from src.components import OxfordAPIResponse
from typing import List


class OxfordDictAPI():
    # write a detailed comment on the OxfordDictAPI class
    """
    Wrapper class for the Oxford Dictionary API. It retrieves the senses and examples for a given word.

    Attributes:
    ----------
        word_id: str
            The word we want the examples for
    """
    def __init__(
            self,
            word_id: str
    ):
        if not isinstance(word_id, str):
            raise ValueError(
                f'Expected word_id to be a string, but got {type(word_id)}'
            )

        self.loggig = logging.basicConfig(level='INFO')
        self.api_creds = OxfordAPISettings()
        self.word = word_id
        self.query = ('entries', 'sentences')
        self.url_ent = f'{self.api_creds.url}/{self.query[0]}/en/'
        self.url_sent = f'{self.api_creds.url}/{self.query[1]}/en/'
        self.strict_match = f'?strictMatch={self.api_creds.strict_match}'

        self.url_entries = self.url_ent + self.word + self.strict_match
        self.url_sentences = self.url_sent + self.word + self.strict_match

        self.res_entries = requests.get(
            self.url_entries,
            headers={'app_id': self.api_creds.app_id, 'app_key': self.api_creds.app_key}
        )

        self.res_sentences = requests.get(
            self.url_sentences,
            headers={'app_id': self.api_creds.app_id, 'app_key': self.api_creds.app_key}
        )

        self.sentences = None
        self.senses = []
        self.oxford_word = {}

    def _load_into_json(self, res: Response) -> Dict:
        """
        Load the request into json usable object
        Args:
            res: Response object

        Returns:
            A dict with all the Oxford API objects
        """
        res.raise_for_status()
        if not res.status_code == 200:
            raise ValueError(
                f'The API is not responsive, error status code {res.status_code}'
            )
        json_output = json.dumps(res.json())
        return json.loads(json_output)


    def _preprocessing(self, sentence:str, main_word:str) -> str:
        """
        Search the sentence given and change the main word into its stem
        Args:
            sentence:
            main_word: Word with inflections

        Returns:
            The stemmed word version
        """
        words = sentence.split()
        for idx, w in enumerate(words):
            if re.search(main_word[:3], w.lower()):
                words[idx] = main_word
        return ' '.join(words)

    def _yield_component(self) -> Dict:
        """
        Yield's back an OxfordAPI object with:
            - word key: the main word
            - senses: A list of the different senses, definitions and examples from each sense
        Returns:
            An OxfordAPI object
        """
        logging.info(
            f'{"-" * 20} Extracting sentence examples from the Oxford API for the desired word: "{self.word}" {"-" * 20}')

        # Load the response into json
        # Create the senses and sentences objects
        self.senses_examples = self._load_into_json(self.res_entries)
        self.sentences_examples = self._load_into_json(self.res_sentences)

        if ('results' not in self.senses_examples.keys()) or ('results' not in self.sentences_examples.keys()):
            raise ValueError(
                f'No results from the Oxford API for the word: "{self.word}"'
            )

        # Create a list of all the senses by iterating over the sentences object
        senses_all_res = self.senses_examples['results']
        sentences_all_res = self.sentences_examples['results']

        sense_with_examples = {}
        diff_sense_ids = []

        # Iterate over the sentences object and create a list of the different sense ids
        for res_s in sentences_all_res:
            for ent in res_s['lexicalEntries']:
                for el in ent['sentences']:
                    diff_sense_ids.append(el['senseIds'][0])

        sense_ids = set(diff_sense_ids)

        # Return the sentences for a given sense id
        def search(id):
            for res_s in sentences_all_res:
                for ent in res_s['lexicalEntries']:
                    return [self._preprocessing(sent['text'], self.word) for sent in ent['sentences'] if
                            sent['senseIds'][0] == id]

        # Iterate over the senses object and create a list of the different senses, definitions and examples
        # Yield the OxfordAPI object
        for res in senses_all_res:
            for lent in res['lexicalEntries']:
                for ent in lent['entries']:
                    for idx, sens in enumerate(ent['senses']):
                        try:
                            if not 'examples' in sens.keys():
                                continue

                            sense_with_examples['id'] = sens['id']
                            sense_with_examples['definition'] = sens['definitions'][0]
                            examples_for_senses = list(self._preprocessing(ex['text'], self.word) for ex in sens['examples'])

                            if sens['id'] in list(sense_ids):
                                examples_sense = search(sens['id'])
                                sense_with_examples['examples'] = list(chain(examples_sense, examples_for_senses))
                            else:
                                sense_with_examples['examples'] = examples_for_senses

                        except KeyError:
                            raise ValueError(
                                'No examples for the word: {}'.format(self.word)
                            )

                        try:
                            yield OxfordAPIResponse(**sense_with_examples).dict().copy()
                        except ValueError:
                            continue

    def get_senses(self) -> Dict:
        """
        Retrieve all senses and examples for a given word
        Returns:
            A dict with the word and the senses
        """
        self.oxford_word['word'] = self.word
        self.oxford_word['senses'] = list(self._yield_component())
        if self.oxford_word['senses'] == []:
            raise ValueError(
                f'No available senses for the word {self.word}'
            )
        return self.oxford_word



if __name__ == '__main__':
    print(OxfordDictAPI('struggle').get_senses())

    
    

    
    
    
    





    