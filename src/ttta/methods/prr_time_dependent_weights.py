import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np
from pathlib import Path
import pickle
from tqdm.autonotebook import trange, tqdm
from scipy.optimize import minimize
from .prr_time_independent_weights import PoissonReducedRankTimeIndependentWordWeights, PoissonReducedRankParameters
from typing import Optional

class PenalizedPoissonReducedRankTimeDependentWordWeights(BaseEstimator):
    """This model is the poisson reduced rank model with time dependent word
    weights. In this model, the word weights are allowed to depend on time t.
    The model is optimized in two steps. First, the model runs through an
    identification step in which the fussed lasso problem in eq. 5 is solved
    (constrained model/log-likelihood optimization). In the second step, the
    model is optimized in the estimation step (unconstrained log-likelihood)
    optimization.

    Note:
        - Currently, only balanced data is supported, i.e. each document is available for the same time indices t.
        - For now, the last term in eq. 5, involving the time lags of word weights, is not implemented. Optimization of the constrained model only happens on the first two terms in eq.5.
    """

    def __init__(self, I: int, T: int, J: int, K: int = 2, **kwargs):
        """Initializes a Penalized Reduced Rank Model with time dependent word
        weights.

        Args:
            K (int, optional): Number of word vector dimensions. Defaults to 2.
        """
        self.k = K
        self.I = I
        self.T = T
        self.J = J

        # define parameters
        self.alpha = np.empty((self.J))
        self.beta = np.empty((self.I * self.T))
        self.b = np.empty((self.k, self.J, self.T))
        self.f = np.empty((self.k, self.I, self.T))
        self.roh1 = np.empty((self.J, self.k))
        self.roh2 = np.empty((self.J, self.k))
        self.X: Optional[np.ndarray] = None

        ## initialize hyper parameters
        self.hyp_roh: Optional[np.ndarray] = None
        self.hyp_delta = None

        # params of the first starting model
        self.params_start_model: Optional[PoissonReducedRankParameters] = None

        # process possible hyperparamteres in **kwargs
        if "roh" in kwargs.keys():
            if isinstance(kwargs["roh"], np.ndarray) == False:
                raise ValueError(
                    "Keyword argument roh must be of type np.ndarray (i.e. created with np.array)"
                )
            if kwargs["roh"].shape != (K,):
                raise ValueError("roh must be of shape (K,)")
            if np.all(kwargs["roh"] > 0) == False:
                raise ValueError("All roh values must be greater 0")

            self.hyp_roh = kwargs["roh"]
        else:
            self.hyp_roh = np.full((self.k,), 0.5)

        if "delta" in kwargs.keys():
            if isinstance(kwargs["delta"], np.ndarray) == False:
                raise ValueError(
                    "Keyword argument delta must be of type np.ndarray (i.e. created with np.array)"
                )
            if kwargs["delta"].shape != (K,):
                raise ValueError("delta must be of shape (K,)")
            if np.all((kwargs["delta"] > 0) & (kwargs["delta"] < 1)) == False:
                raise ValueError("All delta values must be in the open interval (0,1)")

            self.hyp_delta = kwargs["delta"]
        else:
            self.hyp_delta = np.full((self.k,), 0.5)

    @staticmethod
    def calculate_IJT(
        texts: pd.DataFrame,
        text_column: str = "text",
        date_column: str = "date",
        individual_column: str = "individual",
    ) -> dict:
        """Calculates I, J, T for a given dataframe, which can be used for the
        model initialization.

        Args:
            texts (pd.DataFrame): Dataframe containing the data
            text_column (str, optional): column name containing the texts. Defaults to "text".
            date_column (str, optional): column name containing the date. Defaults to "date".
            individual_column (str, optional): column name containing the individual. Defaults to "individual".

        Returns:
            dict: Dictionary with keys I, T, J
        """
        stats = {}
        stats["I"] = len(texts.loc[:, individual_column].unique())
        stats["T"] = len(texts.loc[:, date_column].unique())

        texts_list = sum(texts[text_column].to_list(), [])
        vocab = list(dict.fromkeys(texts_list))
        stats["J"] = len(vocab)

        return stats

    @staticmethod
    def _compute_roh1_matrix(
        roh: np.ndarray, delta: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Computes the parameter matrix roh_{1,j}^k used in the fussed lasso
        optimization problem.

        Args:
            roh (np.ndarray): hyper parameter used for the optimization of the fussed lasso problem. Must be of shape (k,)
            delta (np.ndarray): hyper parameter used for the optimization of the fussed lasso problem. Must be of shape (k,)
            b (np.ndarray): word weights of the poisson reduced rank method with independent word weights. Must be of the shape (J, k)

        Returns:
            np.ndarray: computed matrix for roh_1. same shape as input b.
        """
        numerator = roh * delta
        denominator = np.abs(b)
        return numerator / denominator

    @staticmethod
    def _compute_roh2_matrix(
        roh: np.ndarray, delta: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Computes the parameter matrix roh_{2,j}^k used in the fussed lasso
        optimization problem.

        Args:
            roh (np.ndarray): hyper parameter used for the optimization of the fussed lasso problem. Must be of shape (k,)
            delta (np.ndarray): hyper parameter used for the optimization of the fussed lasso problem. Must be of shape (k,)
            b (np.ndarray): word weights of the poisson reduced rank method with independent word weights. Must be of the shape (J, K)

        Returns:
            np.ndarray: computed matrix for roh_2
        """
        numerator = roh * (1 - delta)
        denominator = np.abs(b)
        return numerator / denominator

    def _convert_params_2Dto3D(
        self, alpha: np.ndarray, beta: np.ndarray, b: np.ndarray, f: np.ndarray
    ) -> None:
        """Converts the parameters from the first step [S1], which are in 2D
        matrices to 3D parameter matrices.

        Args:
            alpha (np.ndarray): array containing a_j
            beta (np.ndarray): array containing b_it
            b (np.ndarray): matrix of the form (J_tokens,K)
            f (np.ndarray): matrix of the form (K, L_documents)

        Returns:
            None: Overrides the class intern parameters to 3D matrices
        """
        self.alpha = alpha
        self.beta = beta
        self.b = (
            b.transpose().reshape(self.k, self.J, 1).repeat(repeats=self.T, axis=2)
        )  # (K,J,T)
        self.f = f.transpose().reshape(self.k, self.I, self.T)  # tranpose(0,2,1)

    @staticmethod
    def get_log_likelihood(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Retrieves the log likelihood matrix, where each element is computed
        by -mu + y*log(mu)

        Args:
            y (np.ndarray): matrix containing the real token counts. Expected shape (J, I*T)
            mu (np.ndarray): matrix containing the mu values. Expected shape (J, I*T)

        Returns:
            np.ndarray: matrix containing the computed log-likelihood for each element. The total sum of can be computed via np.sum
        """
        assert y.shape == mu.shape, "y and mu must be of same shape"
        return -mu + np.multiply(y, np.log(mu))

    def _fussed_lasso_computation(
        self, roh: np.ndarray, matrix: np.ndarray
    ) -> np.float32:
        """Function can be used for the fussed lasso terms of the model.

        Args:
            roh (np.ndarray): parameter matrix of the the form (J,K)
            matrix (np.ndarray): matrix to compute the sums over. Must be of shape (K,J,T).

        Returns:
            np.float32: float containing the sum of all matrix elements.
        """
        assert roh.shape == (matrix.shape[1], self.k), "roh mst be of shape (J,K)"
        # reshape roh to get (K,J,1)
        roh = roh[..., np.newaxis].transpose(1, 0, 2)
        roh_matrix = np.multiply(roh, np.abs(matrix))
        sum_jt = np.sum(roh_matrix, axis=(1, 2))
        sum_jt_k = np.sum(sum_jt)
        return sum_jt_k

    def _obj_function_penalized(
        self,
        y: np.ndarray,
        roh1: np.ndarray,
        roh2: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        b: np.ndarray,
        f: np.ndarray,
    ) -> np.float32:
        """Computes the sum of the loss function for the fussed lasso problem
        given in formula (5) of the paper.

        Args:
            y (np.ndarray): matrix of y token counts. Must be of the form (J, T*I)
            roh1 (np.ndarray): roh1 matrix used for the first term. Must be of the form (J,K)
            roh2 (np.ndarray): roh matrix used for the second (time-differenced) term. Must be of the form (J,K)
            alpha (np.ndarray): array with J elements
            beta (np.ndarray): array with I*T elements
            b (np.ndarray): b matrix containing b_jt^(k). Must be of the form (K,J,T).
            f (np.ndarray): f matrix containing b_it^(k). Must be of the form (K,I,T).

        Returns:
            np.float32: positive Value of the computed loss function.
        """

        mu = self.get_mu(b=b, f=f, alpha=alpha, beta=beta)
        neg_ll = -1 * np.sum(self.get_log_likelihood(y, mu))
        # calculate fussed lasso terms
        term1 = self._fussed_lasso_computation(roh1, b)
        # time differenced term
        b_diff = np.diff(b, n=1, axis=2)  # axis 2 is the time axis.
        term2 = self._fussed_lasso_computation(roh2, b_diff)

        # calculate penalized part
        return neg_ll + term1 + term2

    @staticmethod
    def get_mu(
        b: np.ndarray,
        f: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
    ):
        """Uses the input parameters of the model to compute the mu matrix for
        the Poisson Matrix.

        Args:
            b (np.ndarray): parameter matrix containing b_jt. Must be of form (K, J_tokens, T_timepoints)
            f (np.ndarray): parameter matrix containing f_it. Must be of dimensions (K, I_documents, T_timepoints)
            alpha (Optional[np.ndarray], optional): Vector containing a_j. Must be of dimensions (J_tokens,). Defaults to None.
            beta (Optional[np.ndarray], optional): Vector containing b_it. Must be of dimensions (I*T,) Defaults to None.

        Returns:
            np.ndarray: returns an np.ndarray of the form (J,T*I), where the first T columns correspond to T = 1 for all individuals
        """
        assert b.shape[2] == f.shape[2], "Time dimensions for b and f not equal"
        assert b.shape[0] == f.shape[0], "Dimension k not equal for b and f"

        res = []
        for i in range(b.shape[2]):
            # print(i)
            res.append(np.tensordot(b[:, :, i], f[:, :, i], axes=((0), (0))))
            # print(np.tensordot(b[:,:,i],f[:,:,i],axes=((0),(0))))

        concat_array = np.concatenate(res, axis=1)
        # add alpha
        if alpha is not None:
            concat_array = np.add(alpha.reshape(-1, 1), concat_array)
        if beta is not None:
            concat_array = np.add(beta.reshape(1, -1), concat_array)

        return concat_array

    def fit(
        self,
        texts: pd.DataFrame,
        text_column: str = "text",
        date_column: str = "date",
        individual_column: str = "individual",
        n_iter: int = 100,
        train_params: dict = {"prrr_indep_niter": 5},
        tol: float = 1e5,
        warm_start: Optional[PoissonReducedRankParameters] = None,
    ):
        """Fits a Poisson Reduced Rank model with time dependent word weigths
        to the given term-document-matrix (tdm). The optimization routine
        implemented in this paper is described in Jentsch et. al. 'Time-
        dependent Poisson reduced rank models for political text data analysis'
        section 2.3. For the unpenalized/unconstrained model, please see the
        class UnconstrainedPoissonReducedRankTimeDependentWordWeights.

        Args:
            texts (pd.DataFrame): A pandas DataFrame containing the columns text_column, date_column and individual_column containing the documents and their respective dates. The column combination of individual_column and date_column leads to a panel like data structure.
                   The dates must be in a format interpretable by pandas.to_datetime(). Each element of the text_column column must be a list of strings. Internally, the create_dtm function is used with no control over the vocab argument of the create_dtm function currently.
            text_column (str): The name of the column in texts containing the documents. Defaults to "text".
            date_column (str): The name of the column in texts containing the dates. Defaults to "date".
            individual_column (str): An id column, containing the ids (ints) of the individual. Defaults to "individual".
            n_iter (int, optional): Maximum number of iterations for the optimization of the model. Defaults to 100.
            tol (float, optional): Currently not implemented. Tolerance for no improvement. If the difference between the likelihoods of the previous and last iteration is smaller tol, the optimization is stopped. Defaults to 1e5.
            train_params (dict): Additional parameters for the training process. prrr_indep_niter controlls the number of iterations for the independent word weights model used to as a starting point. prrr_indep_niter must be present.
        """
        # TODO: Implement possibility that parameters from previous model can be used.
        self.df = texts.copy()

        if warm_start:
            print("Using params from warm start...")
            current_params = warm_start
            model = PoissonReducedRankTimeIndependentWordWeights(self.k)

        else:
            print("Calculate initial values for the estimation...")

            # [S1] get initial values for the estimation
            model = PoissonReducedRankTimeIndependentWordWeights(self.k)

            current_params = model.fit(
                texts,
                text_column=text_column,
                date_column=date_column,
                individual_column=individual_column,
                n_iter=train_params["prrr_indep_niter"],
            )

        self.X = model._df2dtm(
            texts,
            text_column=text_column,
            date_column=date_column,
            individual_column=individual_column,
        )  # returns (J_tokens, L_documents) dtm matrix

        self.params_start_model = current_params

        print("Convert initial parameters to right dimensions")
        # convert the parameters from the poisson reduced rank model to the params of this models.
        # In particular f must be reshaped and the parameters b_j must be duplicated for all time points t.
        self._convert_params_2Dto3D(
            current_params.alpha,
            current_params.beta,
            current_params.b,
            current_params.f,
        )

        # compute roh 1 and roh 2.
        self.roh1 = self._compute_roh1_matrix(
            self.hyp_roh, self.hyp_delta, current_params.b
        )
        self.roh2 = self._compute_roh2_matrix(
            self.hyp_roh, self.hyp_delta, current_params.b
        )

        print("Optimize model...")
        # [S2] Repeat (i) - (v) below until a convergence criterion is met for steps m=1,....,M
        for i in trange(n_iter):
            # (i) set current params from the intial estimation (ommitted here, just use current_params)

            # (ii) fix current values alpha_j and b_jt and update beta and f by maximizing the poisson log likelihood (done internally)
            self._update_f_and_beta()

            #  (iii) fix the beta and f from from the previous run/inital params and update alpha and b solving for the fused lasso problem (see formula 5 in paper)
            self._update_b_and_alpha()

            # (iv) + (v) Reparameterize and override the current params
            self._reparameterize()

        return current_params

    def _reparameterize(self):
        """Reparameterizes the variables beta, alpha, b and f  according to
        step [S2] (iv)"""
        self.alpha = self.alpha + np.sum(self.beta) / self.I * self.T
        self.beta = self.beta - np.sum(self.beta) / self.I * self.T
        self.b = self.b * np.sqrt(np.sum(self.f**2) / self.I * self.T)
        self.f = self.f / np.sqrt(np.sum(self.f**2) / self.I * self.T)

    def _update_f_and_beta(self):
        """Updates the parameter vectors f and beta for the identification of
        th emodel.

        Args:
            y (np.ndarray): y token matrix of the form (J,I*T)
        """

        def wrapper_optim_f_and_beta_given_alpha_and_b(
            beta_and_f: np.ndarray, *args
        ) -> np.float32:
            """Function used as a wrapper function in scipy optimize. Since
            scipy.optimize only takes 1D arrays.

            Args:
                beta_and_f (np.ndarray): flattened np.ndarray. The first I*T values correspond to the vector beta and the remaining values to f. F is reshaped for computation of the loss.
                *args (Tuple): Tuple of the form (y,alpha,b, len_beta, K, I, J, T) the values for y, alpha and b. No reshaping is needed, just pass them in a tuple as they are

            Returns:
                np.float32: Loss of negative log likelihood which is minimized be the scipy.minimize function. Just the matrix sum of the log-likelihood matrix.
            """

            # unpack *args arguments (y,alpha,b) expected
            y = args[0]
            alpha = args[1]
            b = args[2]
            len_beta = args[3]
            K = args[4]
            I = args[5]
            J = args[6]
            T = args[7]

            # get beta
            beta1D = beta_and_f[:len_beta]
            f1D = beta_and_f[len_beta:]
            # reshape f to get matrix for get_mu
            f = f1D.reshape(K, I, T)

            # note: since the wrapper function is enclosed by a function with acces to self, I do not need to pass self
            mu = self.get_mu(b=b, f=f, alpha=alpha, beta=beta1D)
            return -1 * np.sum(self.get_log_likelihood(y, mu))

        ## flatten current params
        f_lattened = self.f.flatten(order="C")
        beta = self.beta.flatten()
        len_beta = beta.size

        concat_beta_and_f = np.concatenate([beta, f_lattened], axis=0)

        optim_result = minimize(
            wrapper_optim_f_and_beta_given_alpha_and_b,
            x0=concat_beta_and_f,
            args=(self.X, self.alpha, self.b, len_beta, self.k, self.I, self.J, self.T),
        )

        # unpack results from the optimization
        new_beta, new_f = optim_result.x[:len_beta], optim_result.x[len_beta:]

        # update internal params with new values
        self.f = new_f.reshape(self.f.shape)
        self.beta = new_beta.reshape(self.beta.shape)

    def _update_b_and_alpha(self):
        """Updates the internal parameters alpha and b given f and beta.

        Args:
            y (np.ndarray): token matrix of the form (J, I*T)
        """

        def wrapper_optim_b_and_alpha_given_beta_and_f(
            alpha_and_b: np.ndarray, *args
        ) -> np.float32:
            """Function used as a wrapper function in scipy optimize. Since
            scipy.optimize only takes 1D arrays.

            Args:
                alpha_and_b (np.ndarray): flattened np.ndarray. The first J values correspond to the vector alpha and the remaining values to the flattend b. b is reshaped for computation of the loss.
                *args (Tuple): Tuple of the form (y,beta,f, len_alpha, K, I, J, T, roh1, roh2). No reshaping is needed as it is done internally, just pass them in the tuple as they are.

            Returns:
                np.float32: Value for the penalized loss function to optimize for
            """

            # unpack *args arguments (y,alpha,b) expected
            y = args[0]
            beta = args[1]
            f = args[2]
            len_alpha = args[3]
            K = args[4]
            I = args[5]
            J = args[6]
            T = args[7]
            roh1 = args[8]
            roh2 = args[9]

            # get alpha
            alpha = alpha_and_b[:len_alpha]
            b1D = alpha_and_b[len_alpha:]
            # reshape f to get matrix for get_mu
            b = b1D.reshape(K, J, T)

            # note: since the wrapper function is enclosed by a function with acces to self, I do not need to pass self

            loss = self._obj_function_penalized(
                y=y, roh1=roh1, roh2=roh2, alpha=alpha, beta=beta, b=b, f=f
            )
            return loss

        ## flatten current params
        b_flattened = self.b.flatten(order="C")
        alpha = self.alpha.flatten()
        len_alpha = alpha.size

        concat_beta_and_f = np.concatenate([alpha, b_flattened], axis=0)

        optim_result = minimize(
            wrapper_optim_b_and_alpha_given_beta_and_f,
            x0=concat_beta_and_f,
            args=(
                self.X,
                self.beta,
                self.f,
                len_alpha,
                self.k,
                self.I,
                self.J,
                self.T,
                self.roh1,
                self.roh2,
            ),
        )

        # unpack results from the optimization
        new_alpha, new_b = optim_result.x[:len_alpha], optim_result.x[len_alpha:]

        # override internal params with updated values
        self.b = new_b.reshape(self.b.shape)
        self.alpha = new_alpha.reshape(self.alpha.shape)

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

    def fit_update(self):
        raise NotImplementedError

    def infer_vector(self):
        raise NotImplementedError

    def plot_word_weights(self, params: PoissonReducedRankParameters):
        raise NotImplementedError

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
