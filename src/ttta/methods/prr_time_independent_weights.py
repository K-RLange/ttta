import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.utils.extmath import randomized_svd
from pathlib import Path
import pickle
from tqdm.autonotebook import trange, tqdm
from typing import NamedTuple
import numpy.typing as npt
from scipy.optimize import minimize
from ..preprocessing.preprocess import create_dtm


class PoissonReducedRankParameters(NamedTuple):
    """Parameter class used for storing the parameters of the poisson reduced
    rank model.

    Given a tdm of dimensions (J_tokens, L_documents), the dimensions of
    the other values are described as follows alpha has dimensions
    (J_tokens,) beta has dimensions (L_documents,) b has dimensions
    (J_tokens, K) with K being equal to the dimensions of the political
    space f has diemensions (K, L_documents) with K being equal to the
    dimensions of the political space
    """

    logged_dtm: np.ndarray
    alpha: npt.NDArray[np.float32]
    beta: npt.NDArray[np.float32]
    b: np.ndarray
    f: np.ndarray
    svd_u: np.ndarray
    svd_v: np.ndarray
    svd_s: np.ndarray


def poisson_log_likelihood(Y: np.ndarray, mu: np.ndarray) -> np.float32:
    """Computes the log likelihood over two arrays. The implementation ignores
    the factorial of y since this has no effect on the maximization and can be
    ommitted for optimization.

    Args:
        Y (np.ndarray): Matrix or vector of y values (in this package: poisson distributed word counts)
        mu (np.ndarray): mean parameter matrix of poisson parameter

    Returns:
        np.double: returns the sum of the log likelihood
    """
    assert Y.shape == mu.shape, "Y, and theta must be of same shape"
    return np.sum(-mu + np.multiply(Y, np.log(mu)))


def poisson_log_likelihood_parameterized(
    y: np.ndarray, alpha: np.ndarray, beta: np.ndarray, b: np.ndarray, f: np.ndarray
) -> np.float32:
    """Paramterizes the poisson log likelihood for the model Poisson Reduced
    Rank regression with time independent word weights in order to wrap these
    functions for the scipy optimization.

    Args:
        y (np.ndarray): vector of token counts (n_tokens)
        alpha (np.ndarray): vector of average word_weights (n_tokens)
        beta (np.ndarray): vector of average counts per document (m_doucuments)
        b (np.ndarray): vector of word weights for the j-th token (n_tokens, k_latent space dimensions)
        f (np.ndarray): vector of l-th document specific weights

    Returns:
        np.float32: poisson log likelihood value
    """
    if f.ndim == 2 and b.ndim == 2:
        bf = np.dot(b, f)
        logged_mu = bf + alpha.reshape(-1, 1) + beta
    else:
        bf = np.multiply(b, f)
        logged_mu = bf + alpha + beta  # formula for the logged likelihood
    pois_loglik_calc = poisson_log_likelihood(y, np.exp(logged_mu))
    return pois_loglik_calc


class PoissonReducedRankTimeIndependentWordWeights(BaseEstimator):

    def __init__(self, K=2):
        self.k = K
        self._L_documents = 0
        self._J_tokens = 0
        self.data = None

    def _df2dtm(
        self,
        texts: pd.DataFrame,
        text_column: str = "text",
        date_column: str = "date",
        individual_column: str = "individual",
    ) -> np.ndarray:
        """Converts the input pandas data frame to a document term frequency
        matrix.

        Args:
            texts (pd.DataFrame): A pandas DataFrame containing the columns text_column, date_column and individual_column containing the documents and their respective dates. The column combination of individual_column and date_column leads to a panel like data structure.
                   The dates must be in a format interpretable by pandas.to_datetime(). Each element of the text_column column must be a list of strings. Internally, the create_dtm function is used with no control over the vocab argument of the create_dtm function currently.
            text_column (str): The name of the column in texts containing the documents. Defaults to "text".
            date_column (str): The name of the column in texts containing the dates. Defaults to "date".
            individual_column (str): An id column, containing the ids (ints) of the individual. Defaults to "individual".
        Returns:
            np.ndarray: Returns an np.ndarray for which can be used for the fitting
        """
        if text_column not in texts.columns:
            raise ValueError(
                "Column name given by text_column must be present in data frame columns"
            )
        if date_column not in texts.columns:
            raise ValueError(
                "Column name given by date_column must be present in data frame columns"
            )
        if individual_column not in texts.columns:
            raise ValueError(
                "Column name given by individual_column must be present in data frame columns"
            )
        ## Data in the text_column is unpacked and sorted
        data: pd.DataFrame = texts.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        data.sort_values([individual_column, date_column], inplace=True)
        # add missing document, time rows and fill rows with 0 of token matrix
        texts_list = sum(data[text_column].to_list(), [])
        vocab = list(dict.fromkeys(texts_list))
        ## create dtm from all given texts, since the create_dtm function takes a vocab, currently, all present strings are used as the vocab
        dtm_data = create_dtm(texts=data.loc[:, text_column], vocab=vocab)
        dtm = dtm_data[0]  # extracts dtm matrix from result
        ## convert dtm to dataframe
        df_dtm = pd.DataFrame(dtm.toarray())
        df_dtm.columns = [f"feature_{i}" for i in range(df_dtm.shape[1])]
        df_dtm.set_index(
            data.index, inplace=True
        )  # reset index of the dtm_matrix to match it to the initial dataframe
        ## add dtm matrix columns to dataframe
        data = pd.concat([data, df_dtm], axis=1)
        ## add missing rows for (time, individual) dimension
        missing_rows_added = (
            data.set_index([individual_column, date_column])
            .unstack(fill_value=0)
            .stack()
            .reset_index()
        )
        columns = ["feature" in x for x in missing_rows_added.columns]
        matrix = missing_rows_added.loc[:, columns].transpose()

        ## save data for later viewing in class
        self.data = missing_rows_added.copy()

        return matrix.to_numpy()

    @staticmethod
    def _normalize_matrix(X: npt.NDArray) -> dict:
        """Normalizes a matrix and returns a normalized matrix namedTuple,
        containing average occurences of tokens per token across documents and
        per document across token, as well as the centered document term
        matrix.

        Args:
            X (np.ndarray): matrix of the form (n_tokens, n_documents)

        Returns:
            dict: dictionary containing stats for the normalized matrix
        """
        mean_occurence_per_token = np.mean(
            X, axis=1
        )  # mean occurence of tokens (corresponds to alpha in formula)
        # mean occurence of each token in document
        mean_occurence_per_document = np.mean(
            X, axis=0
        )  # avg. token occurence per document
        matrix_mean = np.mean(X)

        dtm_row_normalized = np.subtract(X, mean_occurence_per_token.reshape(-1, 1))
        dtm_column_normalized = np.subtract(
            dtm_row_normalized, mean_occurence_per_document
        )

        return {
            "centered_matrix": dtm_column_normalized,
            "matrix_mean": matrix_mean,
            "mean_per_row": mean_occurence_per_token,
            "mean_per_column": mean_occurence_per_document,
        }

    def _init_params(self):
        # TODO: rewrite the methods with an internal method
        self.alpha = np.empty(self._J_tokens)
        self.beta = np.empty(self._L_documents)
        self.b = np.empty((self._J_tokens, self.k))
        self.f = np.empty((self._L_documents, self.k))

    def get_params(self) -> dict:
        """Returns the parameters for the poisson reduced rank model with
        independent word weights.

        Returns:
            dict: Dictionary containing the parameter arrays
        """
        return {
            "K": self.k,
            "alpha": self.alpha,
            "beta": self.beta,
            "b": self.b,
            "f": self.f,
        }

    def get_random_params(self) -> dict:
        """Returns random parameters for the poisson reduced rank model with
        independent word weights.

        Returns:
            dict: Dictionary containing the random paramters
        """
        return {
            "K": self.k,
            "alpha": np.random.rand(
                self._J_tokens,
            ),
            "beta": np.random.rand(
                self._L_documents,
            ),
            "b": np.random.rand(self._J_tokens, self.k),
            "f": np.random.rand(self._L_documents, self.k),
        }

    def set_params(self, **params) -> None:
        """Sets the internal parameters of the poisson reduced rank model.

        The **params should contains 'alpha', 'beta', 'b' and 'f'. 'K'
        is not allowed, since it would change the whole metho due to the
        increase in dimensions. The dimensionalities are not checked,
        please make sure that the np.ndarrays are of the correct shape.
        """

        if "alpha" in params:
            self.alpha = params["alpha"]

        if "beta" in params:
            self.beta = params["beta"]

        if "b" in params:
            self.b = params["b"]

        if "f" in params:
            self.f = params["f"]

    def save(self, path: Path) -> None:
        """Saves the Poisson Reduced rank model with independent word weights to the given path
        Args:
            path (Path): Path object to the given save location
        """
        with open(path.absolute(), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> dict:
        with open(path.absolute(), "rb") as f:
            return pickle.load(f)

    def fit(
        self,
        texts: pd.DataFrame,
        text_column: str = "text",
        date_column: str = "date",
        individual_column: str = "individual",
        n_iter: int = 100,
        tol: float = 1e-5,
    ):
        """Fits the Poisson Reduced Rank Model with independent word weights to
        the the given data frame. Internally, the given pandas dataframe is
        converted to a numpy document term frequency matrix.

        Args:
            texts (pd.DataFrame): A pandas DataFrame containing the columns text_column, date_column and individual_column containing the documents and their respective dates. The column combination of individual_column and date_column leads to a panel like data structure.
                   The dates must be in a format interpretable by pandas.to_datetime(). Each element of the text_column column must be a list of strings. Internally, the create_dtm function is used with no control over the vocab argument of the create_dtm function currently.
            text_column (str): The name of the column in texts containing the documents. Defaults to "text".
            date_column (str): The name of the column in texts containing the dates. Defaults to "date".
            individual_column (str): An id column, containing the ids (ints) of the individual. Defaults to "individual".
            n_iter (int, optional): number of iteration for fitting. Defaults to 100.
            tol (float, optional): Currently not implemented. Defaults to 1e-5.

        Returns:
            None: Returns nothing. To acces the final parameters, please see the get_params() method
        """
        if self.k <= 0:
            raise ValueError("Dimensions K must be greater 0")
        if n_iter <= 0:
            raise ValueError("Number of iterations must be greater 0")
        print("Converting Dataframe to dtm matrix...")
        ## convert the pandas dataframe to a dtm matrix
        X = self._df2dtm(
            texts=texts,
            text_column=text_column,
            date_column=date_column,
            individual_column=individual_column,
        )  # X (npt.NDArray[np.int32]): array-like of shape (n_tokens, n_documents).
        print("Conversion from Dataframe to dtm successful!")

        ## initialize basic dimensions from given data
        self._J_tokens, self._L_documents = X.shape[0], X.shape[1]

        print(
            f"Fitting a poisson Reduced Rank Model with time indep. Word Weights for {self._J_tokens} tokens and {self._L_documents} documents"
        )
        self._init_params()

        model_params = self._calculate_starting_values(X)

        for i in trange(n_iter):
            log_likelihood_old = poisson_log_likelihood_parameterized(
                X,
                alpha=model_params.alpha,
                beta=model_params.beta,
                b=model_params.b,
                f=model_params.f,
            )
            update_f_and_beta = self._update_f_and_beta(X, model_params)

            updated_params = PoissonReducedRankParameters(
                logged_dtm=model_params.logged_dtm,
                svd_s=model_params.svd_s,
                svd_u=model_params.svd_u,
                svd_v=model_params.svd_v,
                f=update_f_and_beta.f,
                beta=update_f_and_beta.beta,
                b=model_params.b,
                alpha=model_params.alpha,
            )
            update_b_and_alpha = self._update_b_and_alpha(X, updated_params)

            model_params = update_b_and_alpha  # contains updated parameters
            log_likelihood_new = poisson_log_likelihood_parameterized(
                X,
                alpha=model_params.alpha,
                beta=model_params.beta,
                b=model_params.b,
                f=model_params.f,
            )

            if np.abs(log_likelihood_new - log_likelihood_old) < tol:
                print(
                    "Aborted further updates since the Early Stopping (likelihood) executed."
                )
                break

        reparameterized_params = self._reparameterize_identification(model_params)
        return reparameterized_params

    def _reparameterize_identification(
        self, params: PoissonReducedRankParameters
    ) -> PoissonReducedRankParameters:
        """Reparameterizes the parameters in order to meet the identification
        conditions of (2) and (3) given in the paper.

        Args:
            params (PoissonReducedRankParameters): Current parameters, of the model

        Returns:
            PoissonReducedRankParameters: Reparameterized parameters which fulfill the identification conditions.
        """

        # get the logged m_jl
        log_mu_jl = self._convert_parameters2loggedPoissonParameters(params)
        # How to parameterize centered vector: https://mc-stan.org/docs/2_18/stan-users-guide/parameterizing-centered-vectors.html

        # calculate centered dtm statistics, for svd and calculation of poisson parametization
        centered_dtm_stats = self._normalize_matrix(log_mu_jl)
        alpha = centered_dtm_stats["mean_per_row"]
        beta = centered_dtm_stats["mean_per_column"] - centered_dtm_stats["matrix_mean"]

        # next, we need to calculate b_k and f_k, where k defines the dimension of the political space
        # this is done by decomposing the centered dtm
        centered_theta = np.subtract(np.subtract(log_mu_jl, alpha.reshape(-1, 1)), beta)
        u, s, v = self._wrapper_preset_svd(centered_theta)

        # b_k corresponds to the k-th column
        b = np.divide(np.multiply(s, u), np.sqrt(self._L_documents))
        # f_k corresponds to the k-th column
        f = np.multiply(np.sqrt(self._L_documents), v)

        return PoissonReducedRankParameters(
            logged_dtm=centered_theta,
            alpha=alpha,
            beta=beta,
            f=f,
            b=b,
            svd_v=v,
            svd_u=u,
            svd_s=s,
        )

    def _update_f_and_beta(
        self, X: np.ndarray, model_params: PoissonReducedRankParameters
    ) -> PoissonReducedRankParameters:
        """Updates the f matrix of the poisson reduced rank model, given the
        other vectors/matrices (alpha, beta and b). As stated in the
        supplementary material of the paper, the maximization of the matrix is
        identical to seperatly maximizing the matrices column wise.

        Args:
            Y (np.ndarray): logged word occurence
            logged_param_mat (np.ndarray): Logged parameter matrix of the poisson parameter
        """

        new_f = np.empty(shape=model_params.f.shape, dtype=np.float32)
        new_beta = np.empty(shape=model_params.beta.shape, dtype=np.float32)

        for i_col in range(self._L_documents):
            # update f (given alpha_j's and b_j's)
            init_guess = np.concatenate(
                [model_params.beta[i_col].reshape(-1), model_params.f[:, i_col]]
            )  # params to optimize
            update_f_and_beta_result = minimize(
                PoissonReducedRankTimeIndependentWordWeights.wrapper_update_f_poisson_log_likelihood,
                init_guess,
                args=(model_params.alpha, model_params.b, X[:, i_col]),
                options={"maxiter": 1},
            )
            new_f[:, i_col] = update_f_and_beta_result.x[1:]
            new_beta[i_col] = update_f_and_beta_result.x[0]

        return PoissonReducedRankParameters(
            logged_dtm=model_params.logged_dtm,
            alpha=model_params.alpha,
            b=model_params.b,
            f=new_f,
            beta=new_beta,
            svd_s=model_params.svd_s,
            svd_v=model_params.svd_v,
            svd_u=model_params.svd_u,
        )

    def _update_b_and_alpha(
        self, X: np.ndarray, model_params: PoissonReducedRankParameters
    ) -> PoissonReducedRankParameters:
        """Updates the parameter vectors b and intercept alpha for each j. The
        update is done independent of j and given the latent vector f per
        document as well as the average token count per document beta_l.

        Args:
            X (np.ndarray): The word count matrix of dimensions (J_tokens, L_documents)
            model_params (PoissonReducedRankParameters): Model params named tuple, containing the current parameters.

        Returns:
            PoissonReducedRankParameters: NamedTuple containing the updated current parameters.
        """
        new_b = np.empty(shape=model_params.b.shape, dtype=np.float32)
        new_alpha = np.empty(shape=model_params.alpha.shape, dtype=np.float32)

        for j_token in range(self._J_tokens):
            init_guess = np.concatenate(
                [model_params.alpha[j_token].reshape(-1), model_params.b[j_token, :]]
            )
            update_b_and_alpha_result = minimize(
                PoissonReducedRankTimeIndependentWordWeights.wrapper_update_b_poisson_log_likelihood,
                init_guess,
                args=(model_params.beta, model_params.f, X[j_token, :]),
                options={"maxiter": 1},
            )
            new_b[j_token, :] = update_b_and_alpha_result.x[1:]
            new_alpha[j_token] = update_b_and_alpha_result.x[0]

        new_params = PoissonReducedRankParameters(
            logged_dtm=model_params.logged_dtm,
            alpha=new_alpha,
            b=new_b,
            f=model_params.f,
            beta=model_params.beta,
            svd_s=model_params.svd_s,
            svd_v=model_params.svd_v,
            svd_u=model_params.svd_u,
        )

        return new_params

    def _convert_parameters2loggedPoissonParameters(
        self, parameters: PoissonReducedRankParameters
    ) -> np.ndarray:
        """Uses the calculated parameters to construct a matrix of logged
        poisson parameters.

        Args:
            parameters (PoissonReducedRankParameters): Parameters for the poisson reduced rank model. I.e a PoissonReducedRankParameters object.

        Returns:
            np.ndarray: Matrix of logged poisson parameters
        """
        # INFO: In the r-package, the data was computed differently, (see estim.R line 84-90). However,
        # the calculation there was not clear for me, hence I followed the formula in the original paper.
        # theta = parameters.svd_u.dot(diag_s).dot(parameters.svd_v)
        # add th_rm -> rowmeans(theta - theta_mean), th_cm, but must be converted to matrix colmean(theta-theta_mean), th_m (scalar)
        bf = np.dot(parameters.b, parameters.f)
        bf_added_alpha = np.add(
            bf, parameters.alpha.reshape(-1, 1)
        )  ## add alpha vector to each column
        logged_mu = np.add(
            bf_added_alpha, np.transpose(parameters.beta)
        )  ## add row vector beta to result from prev. step

        return logged_mu

    @staticmethod
    def is_identified_correctly(params: PoissonReducedRankParameters, atol=0.1) -> bool:
        """Tests if the given parameters, meet the identification conditions
        (2) & (3) in section 2.1. of "Poisson reduced-rank models with an
        application to political text data of Jentsch et. al.

        Args:
            params (PoissonReducedRankParameters): Parameters of the fitted model
            atol (float, optional): Absolute difference to the identification result. Defaults to 0.1.

        Returns:
            bool: True, if all identification conditions are fullfilled, otherwise false.
        """
        beta_sum = np.sum(params.beta)
        b_sum = np.sum(params.b, axis=0)
        f_sum = np.sum(params.f, axis=1)

        sum_vector_condition = 0

        # check conditions in (2)
        if np.isclose(beta_sum, [sum_vector_condition], atol=atol) == False:
            print(f"Sum over beta: {beta_sum}")
            return False
        if np.allclose(b_sum, np.zeros(params.b.shape[1]), atol=atol) == False:
            print(f"b sum: {b_sum}")
            return True
        if np.allclose(f_sum, np.zeros(params.f.shape[0]), atol=atol) == False:
            print(f"f sum: {f_sum}")
            return True

        # check conditions in (3)
        # TODO: Implement check

        return True

    def _wrapper_preset_svd(
        self, X: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preset singular value decomposition. For more information about the
        internal method and the result, please see
        sklearn.utils.extmath.randomized_svd.

        Args:
            X (np.ndarray): np.ndarray (usually 2D matrix) to decompose

        Returns:
            tuple[np.ndarray, np.ndarray,np.ndarray]: (Unitary matrix having left singular vectors with signs flipped as columns. ; The singular values, sorted in non-increasing order. ; Unitary matrix having right singular vectors with signs flipped as rows.
        """
        u, s, v = randomized_svd(
            M=X,
            n_oversamples=10,
            n_iter=2,
            n_components=self.k,
            random_state=None,
            **kwargs,
        )
        return (u, s, v)

    def _calculate_starting_values(
        self, X: npt.NDArray[np.int32]
    ) -> PoissonReducedRankParameters:
        r"""Calculates the starting values for the optimization of the
        algorithm. In particular :math:`\alpha^(0), \beta^(0), b^(0), f^(0)`.
        If log(0) is encountered, a warning is suppresed.

        Args:
            X (npt.NDArray[np.int32]): Document Term Matrix (DTM) of dimensions (n_tokens, n_documents).
        """
        # As the analysis of variance decomposition of log mu_jit gives the true parameters, log x_jit is used to compute the initial values.
        with np.errstate(divide="ignore"):
            theta = np.log(X)
        # replace inifinite and not a number values --> i.e log(0)=-2
        theta[~np.isfinite(theta)] = -2

        # calculate centered dtm statistics, for svd and calculation of poisson parametization
        centered_dtm_stats = self._normalize_matrix(theta)
        alpha = centered_dtm_stats["mean_per_row"]
        beta = centered_dtm_stats["mean_per_column"] - centered_dtm_stats["matrix_mean"]

        # next, we need to calculate b_k and f_k, where k defines the dimension of the political space
        # this is done by decomposing the centered dtm
        centered_theta = theta - alpha.reshape(-1, 1) - beta
        u, s, v = self._wrapper_preset_svd(centered_theta)

        # b_k corresponds to the k-th column
        b = np.divide(np.multiply(s, u), np.sqrt(self._L_documents))
        # f_k corresponds to the k-th column
        f = np.multiply(np.sqrt(self._L_documents), v)

        return PoissonReducedRankParameters(
            logged_dtm=theta,
            alpha=alpha,
            beta=beta,
            f=f,
            b=b,
            svd_v=v,
            svd_u=u,
            svd_s=s,
        )

    @staticmethod
    def wrapper_update_f_poisson_log_likelihood(
        params: np.ndarray, *args
    ) -> np.float32:
        """Wrapper function for the poisson log likelihood, which is used for
        the scipy optimization routine. It uses f and beta to as parameters and
        takes the other parameters as given.

        Args:
            params (np.ndarray): parameter vector, which is used for the optimization.
            *args: args is used to process three tuples (alpha, b, y).

        Returns:
            np.float32: poisson log likelihood computed on the given parameters.
        """

        # params contains (beta, f1,f2,...fk)
        alpha, b, y = args
        # b = args[1]

        # construct design matrix
        ones = np.ones(b.shape[0])
        B = np.c_[ones, b]
        # y = args[2]
        logged_mu = alpha + B.dot(params)
        log_likelihood = -1 * poisson_log_likelihood(
            y, np.exp(logged_mu)
        )  # -1 turns maximization into minimization for scipy
        return log_likelihood

    @staticmethod
    def wrapper_update_b_poisson_log_likelihood(
        params: np.ndarray, *args
    ) -> np.float32:
        """Wrapper function used to update the parameter vector b and alpha,
        taking the other parameters as given.

        Args:
            params (np.ndarray): Parameter vector for the optimization. *args is a tuple of length 3 and used to feed the "given" parameters (beta, f, y).

        Returns:
            np.float32: Computed log likeklihood on the given parameters.
        """

        # params contains (alpha_j, bj1, ... bjK)
        beta, f, y = args

        ones = np.ones(f.shape[1])
        F = np.c_[ones, np.transpose(f)]

        logged_mu = beta + F.dot(params)
        log_likelihood = -1 * poisson_log_likelihood(
            y, np.exp(logged_mu)
        )  # -1 turns maximization into minimization
        return log_likelihood

    def fit_update(self):
        raise NotImplementedError

    def infer_vector(self):
        raise NotImplementedError

    def plot_word_weights(self, params: PoissonReducedRankParameters):
        raise NotImplementedError
