"""Implements an automatic time chunk creation for diachronic topic modeling using pandas."""
import pandas as pd
from typing import Union, List
from datetime import datetime
import warnings
import numpy as np
from ..methods.lda_prototype import LDAPrototype
from dateutil.relativedelta import relativedelta
import re

def _get_time_indices(texts: pd.DataFrame, how: Union[str, List[datetime]] = "M", last_date: datetime = None,
                      date_column: str = "date", min_docs_per_chunk: int = 500) -> pd.DataFrame:
    """Create the time indices for the given texts.

    If update is True, the time indices are appended to the existing ones.
    Args:
        texts: A pandas DataFrame containing the documents and their respective dates.
               The dates must be in a format interpretable by pandas.to_datetime(). The texts must be a list of strings.
        how: List of datetime dates indicating the end of time chunks or a string indicating the frequency of the time chunks. Used to create time
             chunks when fixed dates were used for the initial fit. If None, the same time chunk rule as in the initial fit is used.
        last_date: The last date of the texts that were used before. If None, the last date of the texts is used.
        date_column: The name of the column containing the dates in the texts DataFrame.
        min_docs_per_chunk: The minimum number of documents that a time chunk must contain. If a time chunk does not contain at least this many
                            documents, it is combined with the next time chunk.
    """
    if not isinstance(texts, pd.DataFrame):
        raise TypeError("texts must be a pandas DataFrame!")
    if not isinstance(date_column, str):
        raise TypeError("date_column must be a string!")
    if min_docs_per_chunk is None:
        min_docs_per_chunk = 1
    if not isinstance(min_docs_per_chunk, int):
        try:
            if min_docs_per_chunk == int(min_docs_per_chunk):
                min_docs_per_chunk = int(min_docs_per_chunk)
            else:
                raise ValueError
        except ValueError:
            raise TypeError("min_docs_per_chunk must be an integer!")
    if min_docs_per_chunk < 1:
        raise ValueError("min_docs_per_chunk must be a natural number greater than 0!")
    df_index = pd.DataFrame({'chunk_start': range(len(texts))}, index=pd.to_datetime(texts[date_column]))
    if last_date is not None:
        if df_index.index.min() <= last_date:
            raise ValueError("The dates of the new texts chronologically overlap with the text that were already used before!")

    if not isinstance(how, (str, list, np.ndarray)):
        raise TypeError("how must be a string or a list of datetime objects!")
    if isinstance(how, str):
        period_start = df_index.resample(how).first()
    else:
        if df_index.index.max() < how[-1]:
            how = how[:-1]
        if sum([df_index.index.min() > x for x in how]) > 1:
            how = how[(sum([df_index.index.min() > x for x in how]) - 1):]
        chunk_dates = pd.DatetimeIndex(how)
        chunks = chunk_dates.searchsorted(df_index.index, "right")
        period_start = df_index.groupby(chunk_dates[np.where(chunks < 1, 0, chunks - 1)]).first()
        period_start.index.name = date_column

    len_before_na_drop = len(period_start)
    period_start = period_start.dropna()
    rows_to_drop = []
    current_docs_since_last_chunk = 0
    for i, (index, row) in enumerate(period_start[1:].iterrows()):
        current_docs_since_last_chunk += row["chunk_start"] - period_start["chunk_start"].iloc[i]
        if current_docs_since_last_chunk < min_docs_per_chunk and not i == len(period_start) - 2:
            rows_to_drop.append(index)
        else:
            current_docs_since_last_chunk = 0
    period_start = period_start.drop(rows_to_drop)
    if len(period_start) < len_before_na_drop:
        warnings.warn(f"{len_before_na_drop - len(period_start)} time chunks do not fulfill the requirement of containing least {min_docs_per_chunk} texts and are combined "
                      f"into larger time chunks.")
    period_start["chunk_start"] = period_start["chunk_start"].astype(np.uint64)
    period_start = period_start.reset_index()
    return period_start


def how_to_timedelta(how: str) -> relativedelta:
    """
    Converts pandas `how` resampling options into parameters suitable for `timedelta`.

    Args:
        how (str): The resampling option, e.g., 'S' for seconds, 'min' for minutes, etc.

    Returns:
        timedelta: A timedelta object corresponding to the `how` parameter.
    """
    mapping = {
        'S': 'seconds',
        'T': 'minutes',
        'min': 'minutes',
        'H': 'hours',
        'D': 'days',
        'W': 'weeks',
        'M': 'months',
        'ME': 'months',
        'Q': 'quarters',
        'Y': 'years',
    }
    match = re.match(r"(\d+)?([a-zA-Z]+)", how)
    if not match:
        raise ValueError(f"Invalid `how` format: {how}. Expected format like '3S', '5min', etc.")
    quantity, unit = match.group(1), mapping[match.group(2)]
    if quantity is None:
        quantity = 1
    else:
        quantity = int(quantity)
    if unit == "quarters":
        quantity *= 3
        unit = "months"
    return relativedelta(**{unit: quantity})
