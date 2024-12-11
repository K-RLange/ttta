import pandas as pd
from .methods.lda_prototype import LDAPrototype
from .methods.rolling_lda import RollingLDA
from .preprocessing import chunk_creation
import warnings
from typing import Union, List, Tuple, Callable
from datetime import datetime
from .preprocessing.preprocess import preprocess


class TTTAPipeline():
    def __init__(self, corpus: Union[pd.DataFrame, str] = None, model: str = None, how: Union[str, List[datetime]] = "M", preprocessing: callable = None,
                 model_save_path: str = None, text_column: str = "text", date_column: str = "date", **kwargs) -> None:
        if corpus is not None:
            if isinstance(corpus, str):
                self.corpus = self._read_corpus_file(corpus)
            elif isinstance(corpus, pd.DataFrame):
                self.corpus = corpus
            else:
                raise TypeError("corpus must be a pandas DataFrame or a path to a corpus file!")
        else:
            self.corpus = None
            warnings.warn("No corpus file was given. A pipeline object will be created, but not be trained on any data.")
        if preprocessing is not None:
            self.corpus[text_column] = preprocessing(self.corpus)
        else:
            if model != "sense_disambiguation":
                self.corpus[text_column] = preprocess(self.corpus, **kwargs)

        if model is None:
            self.model = None
            warnings.warn("No model was given. A pipeline object will be created, but not be trained on any data.")
        elif model in ["LDA", "LDAPrototype"]:
            self.model = LDAPrototype(**kwargs)
        elif model == "RollingLDA":
            self.model = RollingLDA(how=how, **kwargs)
        elif model == "PoiRR":
            self.model = None
            warnings.warn("PoiRR is not yet implemented!")
        elif model == "diachronic_alignment":
            self.model = None
            warnings.warn("diachronic_alignment is not yet implemented!")
        elif model == "sense_disambiguation":
            self.model = None
            warnings.warn("sense_disambiguation is not yet implemented!")
        else:
            raise ValueError("model not recognized!")
        self.model.fit(self.corpus, text_column=text_column, date_column=date_column, **kwargs)
        if model_save_path is not None:
            self.model.save(model_save_path)

    def _read_corpus_file(self, path):
        """Reads a corpus file and returns a pandas DataFrame with the columns
        "date" and "text".

        Args:
            path: path to the corpus file
        Returns:
            pandas DataFrame with the columns "date" and "text"
        """
        if path.endswith('.csv', 'tsv'):
            df = pd.read_csv(path)
        elif path.endswith('.json'):
            df = pd.read_json(path)
        elif path.endswith('.pickle', 'pkl'):
            df = pd.read_pickle(path)
        elif path.endswith('.xml'):
            df = pd.read_xml(path)
        elif path.endswith('.hdf'):
            df = pd.read_hdf(path)
        elif path.endswith('.sql'):
            df = pd.read_sql(path)
        elif path.endswith('.xlsx', "xls"):
            df = pd.read_excel(path)
        else:
            raise ValueError(f'Unsupported filetype: {path.split(".")[-1]}')
        return df