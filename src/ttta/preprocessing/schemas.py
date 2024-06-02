from pydantic import BaseModel, field_validator
from typing import List, Optional
from dataclasses import dataclass


class OxfordAPIResponse(BaseModel):
    id: str = None
    definition: str = None
    examples: Optional[List[str]] = None

    class Config:
        allow_population_by_field_name = True

    def __call__(self, **kwargs):
        self.id = kwargs['id']
        self.definition = kwargs['definition']
        self.examples = kwargs['examples']

        return self

    @field_validator('examples')
    def min_len_examples(cls, v):
        if not len(v) >= 1:
            raise ValueError(
                f'Not Enough examples to compile, given: {len(v)}, expected at least 1'
            )
        return v[:10]


@dataclass(init=True, frozen=True, repr=True)
class Words:
    word: str
    senses: List[OxfordAPIResponse]


class WordSimilarities(BaseModel):
    word: str
    year: int
    sense_ids: List[str]
    props: List[float]


class WordFitted(BaseModel):
    word: str
    sense: str
    years: List[int]
    props: List[float]
    poly_fit: List[float]


class SenseEmbedding(BaseModel):
    id: str = None
    definition: str = None
    embedding: List[float] = None

    class Config:
        allow_population_by_field_name = True

    def __call__(self, **kwargs):
        self.id = kwargs['id']
        self.definition = kwargs['definition']
        self.embedding = kwargs['embedding']

        return self


@dataclass(init=True, frozen=True, repr=True)
class WordSenseEmbedding:
    id: str
    definition: str
    embedding: List[float]

    def __str__(self):
        return f'Sense Id: {self.id}\nSense Definition: {self.definition}\nSense Embedding: {len(self.embedding)}'


@dataclass(init=True, frozen=True, repr=True)
class SenseEmbedding:
    word: str
    senses: List[WordSenseEmbedding]


class Embedding(BaseModel):
    word: str = None
    sentence_number_index: List[List] = None
    embeddings: List[List] = None

    class Config:
        allow_population_by_field_name = True

    def __call__(self, **kwargs):
        self.word = kwargs['word']
        self.sentence_number_index = kwargs['sentence_number_index']
        self.embeddings = kwargs['embeddings']

        return self
