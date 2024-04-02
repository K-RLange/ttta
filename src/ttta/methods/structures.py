import numpy as np
import numpy.typing as npt
from typing import NamedTuple


class PoissonReducedRankParameters(NamedTuple):
    """Parameter class used for storing the parameters of the poisson reduced rank model.
    Given a tdm of dimensions (J_tokens, L_documents), the dimensions of the other values are described as follows
    alpha has dimensions (J_tokens,)
    beta has dimensions (L_documents,)
    b has dimensions (J_tokens, K) with K being equal to the dimensions of the political space
    f has diemensions (K, L_documents) with K being equal to the dimensions of the political space
    """

    logged_dtm: np.ndarray
    alpha: npt.NDArray[np.float32]
    beta: npt.NDArray[np.float32]
    b: np.ndarray
    f: np.ndarray
    svd_u: np.ndarray
    svd_v: np.ndarray
    svd_s: np.ndarray
