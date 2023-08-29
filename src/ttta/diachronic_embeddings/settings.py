
from pydantic import BaseSettings, Field
from dotenv import load_dotenv
import logging
import json
from pydantic import parse_obj_as
from typing import List
from components import Words, SenseEmbedding

load_dotenv(verbose=True)
class OxfordAPISettings(BaseSettings):
    accept:str = Field(..., env="ACCEPT")
    app_id:str = Field(..., env="APP_ID")
    app_key:str = Field(..., env="APP_KEY")
    url:str = Field(..., env="URL")
    strict_match:str = Field(..., env="STRICT_MATCH")

    def __str__(self):
        return self.accept + '\n' + self.app_id + '\n' + self.app_key + '\n' + self.url + '\n'

    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"


class EmbeddingFiles(BaseSettings):
    poly_words_f:str = '../data/target_words/polysemous.txt'
    oxford_word_senses: str = '../data/target_words/senses_oxford_api.json'
    sense_embeddings: str = '../embeddings/embeddings_for_senses.json'
    embeddings_root_dit:str = '../embeddings'
    bert_model:str = 'bert_model_new'
    years_used = [1980, 1982, 1985, 1987, 1989, 1990, 1995, 2000, 2001, 2002, 2003, 2005, 2008, 2009, 2010, 2012,
                     2013, 2015, 2016, 2017, 2018]

class FileLoader():
    @classmethod
    def load_files(
            cls,
            module:str,
            file_keyword:str=None
        ):
        if not module in ['Similarities', 'ExtractSenseEmbeddings', 'PolynomialFitting']:
            raise ValueError(
                f'The module {module} does not exist'
            )

        logging.basicConfig(level=logging.NOTSET)
        files = EmbeddingFiles()

        with open(files.poly_words_f, 'r') as f:
            words = f.read()

        if module == 'Similarities':
            with open(files.sense_embeddings, 'r') as f:
                logging.info(f'{"-" * 10} Loading the embeddings for senses: {f.name} {"-" * 10}')
                senses_embeds = parse_obj_as(List[SenseEmbedding], json.load(f))

            return senses_embeds, words

        if module == 'ExtractSenseEmbeddings':
            with open(files.oxford_word_senses) as f: all_words = parse_obj_as(List[Words], json.load(f))
            return all_words

        if module == 'PolynomialFitting':
            if not file_keyword in words.split('\n'):
                raise ValueError(
                    f'The specified keyword: "{file_keyword}" does not apply to any similarity embedding file'
                )

            with open(f'../embeddings_similarity/embeddings_sim_{file_keyword}.json') as f:
                word_props = json.load(f)

            return word_props, words


if __name__ == '__main__':
    print(FileLoader.load_files('Similarities'))







