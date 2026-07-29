# ugme/_ugme.py
from typing import Optional, Sequence, Tuple, Union
import numpy as np

try:
    from deeptime.base import Estimator, Model
except ImportError:
    class Model:
        def __init__(self):pass
    class Estimator:
        def __init__(self, model=None):
            self._model = model
        def fetch_model(self): return self._model
        def fit_fetch(self, data, **kw): return self.fit(data, **kw).fetch_model()

def _as_tnn(tpm: np.ndarray) -> np.ndarray:
    """Ensure TPM time-series has shape (T, n, n)

    Parameters
    ----------
    tpm : np.ndarray
        A 3-D array of transition probability matrices in one of the layouts
        (T, n, n), (n, n, T), or (n, T, n), where T is the number of time frames
        and n the number of states (dimensions).

    Returns
    -------
    np.ndarray
        The same data reordered to shape (T, n, n) (dtype float).

    Raises
    ------
    ValueError
        If tpm is not 3D, or if no axis can be identified as the time
        axis (e.g. all three axes differ in length).
    """
    tpm = np.asarray(tpm, dtype=float)
    if tpm.ndim != 3:
        raise ValueError(f"Expected shapes (T, n, n), (n, T, n) or (n, n, T), got shape {tpm.shape}.")

    n0, n1, n2 = tpm.shape
    
    # the TPM was already in (T, n, n)
    if n1 == n2:
        return tpm

    # the TPM was (n, n, T), so move T to the front
    elif n0 == n1:
        return np.einsum("ijT->Tij", tpm)

    # the TPM was (n, T, n), so move T to the front
    elif n0 == n2:
        return np.einsum("iTk -> Tik", tpm)

    raise ValueError(f"Check TPM dimensions. Cannot delineate time and dimension axes.")
    
def _column_normalize(tpm: np.ndarray, atol: float = 1e-6) -> Tuple[np.ndarray, str]:
    """Ensure TPMs are column-normalized (transpose row-normalized input)
    Parameters
    ----------
    tpm : np.ndarray
        A 3D array of transition probability matrices with shape (T, n, n),
        where T is the number of time frames and n the number of states.
    atol : float, optional, default=1e-6
        Absolute tolerance for testing whether the row or column sums equal 1.

    Returns
    -------
    tpm : np.ndarray
        The TPMs as column-normalized matrices, shape (T, n, n). Row-normalized
        input is transposed on the last two axes; column-normalized input is
        returned unchanged.
    original : str
        The detected normalization of the input, either "col" or "row".

    Raises
    ------
    ValueError
        If the TPMs are neither row- nor column-normalized to within atol.
    """
    col_sums = tpm.sum(axis=1)
    row_sums = tpm.sum(axis=2)

    if np.allclose(col_sums, 1.0, atol=atol):
        return tpm, "col"
    elif np.allclose(row_sums, 1.0, atol=atol):
        return np.einsum("Tij->Tji", tpm), "row"
    raise ValueError("Check TPMs, they are neither row- nor column-normalized")

def _prepend_identity(tpm: np.ndarray, atol: float=1e-8) -> np.ndarray:
    """Ensure the time-zero TPM is the identity, prepending one if needed

    Parameters
    ----------
    tpm : np.ndarray
        A 3D array of transition probability matrices with shape (T, n, n),
        where T is the number of time frames and n the number of states.
    atol : float, optional, default=1e-8
        Absolute tolerance for testing whether the first frame equals the
        identity.

    Returns
    -------
    np.ndarray
        The TPMs with an identity first frame. If tpm[0] is already the
        identity the input is returned unchanged; otherwise an identity frame
        is prepended, giving shape (T + 1, n, n) and shifting all frame
        indices by one.
    """
    # see if the first element is identity
    n = tpm.shape[1]
    if np.allclose(tpm[0], np.eye(n), atol=atol):
        return tpm

    # if not, then prepend
    else:
        tpm_full = np.empty( (tpm.shape[0] + 1, n, n), dtype=tpm.dtype )
        tpm_full[0] = np.eye(n)
        tpm_full[1:] = tpm
        return tpm_full

def _compute_U(tpm: np.ndarray) -> np.ndarray:
    """Compute the time-local propagator U(t) from a TPM time-series

    Parameters
    ----------
    tpm : np.ndarray
        A 3D array of column-normalized transition probability matrices with
        shape (T, n, n) and tpm[0] equal to the identity, where T is the number
        of time frames and n the number of states.

    Returns
    -------
    np.ndarray
        The time-local propagators, shape (T, n, n), defined by U[0] = I and
        U[t] = TPM(t) [ TPM(t - dt) ]^{-1} for t >= 1.
    """
    T, n, _ = tpm.shape
    U = np.empty_like(tpm)

    # Identity initial condition
    U[0] = np.eye(n)

    # U(t) = TPM(t) [ TPM(t - ∆t) ]^{-1}
    for t in range(1, T):
        U[t] = tpm[t] @ np.linalg.inv( tpm[t-1] )

    return U

def _predict_dynamics(U_inf: np.ndarray, tpm_ref: np.ndarray,
                      tau_R: int, n_steps: int) -> np.ndarray:
    """Propagate the TPM dynamics past tau_R with a constant propagator

    Parameters
    ----------
    U_inf : np.ndarray
        The constant propagator applied for t >= tau_R, shape (n, n). Either
        the bare U(tau_R) or the averaged <U>.
    tpm_ref : np.ndarray
        The reference TPM time-series, shape (T, n, n), used verbatim for
        t < tau_R.
    tau_R : int
        Frame index at which propagation with U_inf begins. Reference frames
        are used before it.
    n_steps : int
        Desired length of the predicted trajectory. Values smaller than the
        reference length are raised to it, so the output is never shorter than
        tpm_ref.

    Returns
    -------
    np.ndarray
        The predicted TPM trajectory, shape (n_steps, n, n), equal to the
        reference for t < tau_R and to [U_inf]^n TPM(tau_R) for
        t = tau_R + n dt.
    """
    # get shapes and allow time to be larger than the reference time
    T, n, _ = tpm_ref.shape
    n_steps = int(max(n_steps, tpm_ref.shape[0]))

    # predicted dynamics
    pred = np.empty( (n_steps, n, n), dtype=tpm_ref.dtype)
    pred[:tau_R] = tpm_ref[:tau_R]

    for k in range(tau_R, n_steps):
        pred[k] = U_inf @ pred[k-1]

    return pred

def _rmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Calculate the root-mean-square error between two TPM trajectories.

    The error is computed over the common time interval of the reference and
    predicted trajectories. At each time point, differences between TPM
    elements are normalized by the number of states, M, before summing over
    matrix elements. The resulting metric is

        RMSE = sqrt((1 / T) * sum_t sum_jk
                    [(reference[t,j,k] - prediction[t,j,k]) / M]**2),

    where T is the number of compared time points and M is the number of
    states.

    Parameters
    ----------
    reference : np.ndarray
        Reference TPM time-series, shape (T_ref, M, M).
    prediction : np.ndarray
        Predicted TPM time-series, shape (T_pred, M, M). If its time dimension
        differs from that of `reference`, only the first
        min(T_ref, T_pred) frames are compared.

    Returns
    -------
    float
        Root-mean-square error between the reference and predicted TPM
        trajectories over their common time interval.
    """
    # make sure the two arrays are the same length in time
    T = min(reference.shape[0], prediction.shape[0])
    ref, pred = reference[:T], prediction[:T]

    # dimension
    M = ref.shape[1]

    # difference -> square -> sum -> sqrt
    diff = (ref - pred) / M
    error = np.sqrt(  (diff**2).sum() / T  )

    return float(error)

class UGMEModel(Model):
    """
    UGME model to store the reference dynamics, U(t), and the predicted dynamics
    (with and without averaging)

    Parameters
    ----------
    tpm : ndarray (T, n, n)
          Column-normalized reference TPMs with TPM[0] == I
    U   : ndarray (T, n, n)
          Time-local propagator U[t] = TPM[t] @ [ TPM[t - ∆t] ]^{-1}
    dt  : float
          TPM frame spacing
    """
    def __init__(self, tpm, U, dt=1.0, original_normalization="col", identity_prepended=False):
        super().__init__()
        self._tpm = tpm
        self._U = U
        self._dt = float(dt)
        self._original_normalization = original_normalization
        self._identity_prepended = identity_prepended

    # so I don't need parentheses for calling these things
    @property
    def tpm(self): return self._tpm
    @property
    def U(self): return self._U
    @property
    def n_states(self): return self._tpm.shape[1]
    @property
    def n_steps(self): return self._tpm.shape[0]
    @property
    def dt(self):return self._dt
    @property
    def original_normalization(self): return self._original_normalization
    @property
    def identity_prepended(self): return self._identity_prepended

    
    def check_initial_U(self, atol=1e-8) -> bool: 
        """Check that the initial time-local propagators satisfy the expected identities.

        Verifies that U[0] is the identity matrix and that U[1] is equal to the
        first nonzero-time reference TPM. These conditions follow from the
        initialization convention TPM[0] = I and the definition of the time-local
        propagator.

        Parameters
        ----------
        atol : float, optional
            Absolute tolerance used in the numerical comparisons. Default is 1e-8.

        Returns
        -------
        bool
            True if U[0] is equal to the identity matrix and U[1] is equal to
            TPM[1] within the specified tolerance; False otherwise.
        """
        eye = np.eye(self.n_states)
        return ( np.allclose(self._U[0], eye, atol=atol)
                and np.allclose(self._U[1], self._tpm[1], atol=atol) )

    def reconstruct_tpm(self) -> np.ndarray:
        """Reconstruct the TPM trajectory from the time-local propagators.

        Starting from TPM[0] = I, recursively applies the time-local propagators
        according to

            TPM[t] = U[t] @ TPM[t - 1]

        for each subsequent time point. This provides a consistency check that the
        stored propagators reproduce the reference TPM dynamics.

        Returns
        -------
        np.ndarray
            Reconstructed TPM time-series with the same shape and dtype as the
            reference TPM array, `(T, n, n)`.
        """
        test = np.empty_like(self._tpm)
        test[0] = np.eye(self.n_states)
        for t in range(1, self.n_steps):
            test[t] = self._U[t] @ test[t - 1]
        return test

    def predict(self, tau_R: int, n_steps: Optional[int] = None) -> np.ndarray:
        """Predict TPM dynamics using a constant long-time propagator.

        Uses U(tau_R) as the asymptotic time-local propagator and propagates the
        reference TPM dynamics beyond tau_R according to

            TPM[t] = U(tau_R) @ TPM[t - 1].

        Reference TPMs are retained for times before tau_R. No averaging of the
        time-local propagator is performed.

        Parameters
        ----------
        tau_R : int
            Frame index at which propagation with the constant propagator begins.
            Must satisfy 1 <= tau_R < n_steps of the reference TPM trajectory.
        n_steps : int, optional
            Desired length of the predicted TPM trajectory. If None, the length of
            the reference TPM trajectory is used. Values shorter than the reference
            trajectory are raised to the reference length by `_predict_dynamics`.

        Returns
        -------
        np.ndarray
            Predicted TPM time-series, shape `(n_steps, n, n)`, equal to the
            reference TPMs for t < tau_R and propagated using U(tau_R) thereafter.

        Raises
        ------
        ValueError
            If tau_R is not between 1 and the final reference frame index.
        """
        if not (1 <= tau_R < self.n_steps):
            raise ValueError(f"Requires 1 ≤ tau_R < {self.n_steps}.")
    
        if n_steps is None:
            n_steps = self.n_steps

        return _predict_dynamics(self._U[tau_R], self._tpm, tau_R, n_steps)

    def rmse(self, tau_R: int) -> float:
        return _rmse(self._tpm, self.predict(tau_R))

    def scan_rmse(self, tau_R_values: Sequence[int]) -> np.ndarray:
        return np.array([self.rmse(int(t)) for t in tau_R_values])

    def average_U(self, t_r: int, tau_R: int) -> np.ndarray:
        """Calculate the time-averaged propagator over a specified interval.

        Averages the time-local propagators U(t) from frame t_r up to, but not
        including, tau_R. The resulting propagator is

            <U> = (1 / (tau_R - t_r)) * sum_{t=t_r}^{tau_R-1} U[t].

        Parameters
        ----------
        t_r : int
            Starting frame index of the averaging window. Must satisfy
            0 <= t_r < tau_R.
        tau_R : int
            End frame index of the averaging window. U[tau_R] is not included
            in the average. Must satisfy tau_R <= the number of reference frames.

        Returns
        -------
        np.ndarray
            Time-averaged propagator, shape `(n, n)`.

        Raises
        ------
        ValueError
            If the averaging interval does not satisfy
            0 <= t_r < tau_R <= n_steps.
        """
        if not (0 <= t_r < tau_R <= self.n_steps):
            raise ValueError(f"require 0 <= t_r < tau_R <= {self.n_steps}")

        # average over time axis
        return self._U[t_r:tau_R].mean(axis=0)

    def predict_averaged(self, t_r, tau_R, n_steps=None) -> np.ndarray:
        n_steps = self.n_steps if n_steps is None else n_steps
        return _predict_dynamics(self.average_U(t_r, tau_R),
                                 self._tpm, tau_R, n_steps)

    def rmse_averaged(self, t_r: int, tau_R: int) -> float:
        return _rmse(self._tpm, self.predict_averaged(t_r, tau_R))

    def select_tau_R(self, threshold=0.05, averaged=False, t_r=0,
                     tau_R_values=None) -> int:
        if tau_R_values is None:
            tau_R_values = np.arange(max(t_r + 1, 1), self.n_steps)
        tau_R_values = np.asarray(tau_R_values, dtype=int)
        if averaged:
            rmses = np.array( [self.rmse_averaged(t_r, int(t)) for t in tau_R_values] )
        else:
            rmses = self.scan_rmse(tau_R_values)

        # threshold
        mask = np.where(rmses <= threshold)[0]
        return int(tau_R_values[mask[0]]) if mask.size else int(tau_R_values[int(np.argmin(rmses))])

class UGME(Estimator):
    r"""generalized master equation (U-GME) estimator

    Validate/check TPM time-series and extract GME with it

    Parameters
    ----------
    dt   : float, optional, default=1.0
           Spacing between TPM frames.
    atol : float, optional, default=1e-6
           Tolerance for the normalization check.
    """
    def __init__(self, dt: float = 1.0, atol: float=1e-6):
        super().__init__()
        self.dt = float(dt)
        self.atol = float(atol)

    def fit(self, data: Union[np.ndarray, str], **kwargs) -> "UGME":
        # cool trick from software class: Union allows for either input type
        tpm = _as_tnn(data)
        tpm, orig = _column_normalize(tpm, atol=self.atol)
        tpm = _prepend_identity(tpm)
        U = _compute_U(tpm)
        self._model = UGMEModel(tpm=tpm, U=U, dt=self.dt,
                                original_normalization=orig)
        return self

    def fetch_model(self) -> Optional[UGMEModel]:
        return getattr(self, "_model", None)

    def fit_fetch(self, data, **kwargs) -> UGMEModel:
        # allow for input tpms to be a string/file path
        if isinstance(data, str):
            data = np.load(data, allow_pickle=True)
        return self.fit(data, **kwargs).fetch_model()
