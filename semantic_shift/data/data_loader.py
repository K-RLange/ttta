
from typing import Union, List
from pathlib import Path
import xml.etree.ElementTree as ET 
from semantic_shift.utils.utils import read_txt, sample_data


class Loader():
    def __init__(
            self,
            texts: List[str]
            ):
        
        self.texts = texts
    

    @classmethod
    def from_txt(cls,
                 path: Union[str, Path]
                 ):
        return cls(read_txt(path))
    
    @classmethod
    def from_xml(cls,
                 path: Union[str, Path],
                 tag: str
                 ):
        
        tree = ET.parse(path)
        root = tree.getroot()
        texts = []
        for elem in root.findall('.//' + tag):
            if isinstance(elem.text, str):
                texts.append(elem.text)
        return cls(texts)
    
    def forward(self, target_words: List[str] = None, max_documents: int = None, shuffle: bool = True, random_seed=None):
        if target_words is not None:
            relevant_texts = []
            for text in self.texts:
                if any([word in text for word in target_words]):
                    relevant_texts.append(text)
            
            self.texts = relevant_texts
        
        if max_documents is not None:
            if shuffle:
                self.texts = sample_data(self.texts, max_documents, random_seed)
            else:
                self.texts = self.texts[:max_documents]

        return self.texts
        

