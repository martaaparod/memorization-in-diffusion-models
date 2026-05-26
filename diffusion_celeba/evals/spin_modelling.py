import torch
import numpy as np


def convert_labels_to_spins(labels):
    """Convert binary labels in {0, 1} to spin configurations in {-1, +1}."""
    return labels * 2 - 1


def fit_ising_to_spins(
    spins: np.ndarray | torch.Tensor,
    log_reg_kwargs={"solver": "lbfgs", "penalty": "l2", "C": 1e1, "max_iter": 1000},
):
    """Fit an Ising model to {-1, +1} spins using node-wise logistic regression.

    Args:
        spins: Array-like of shape (n_samples, d), with values in {-1, +1}.
        log_reg_kwargs: Keyword arguments for sklearn.linear_model.LogisticRegression.

    Returns:
        (h_i_est, J_ij_est): Estimated external fields and symmetric couplings.
    """
    if hasattr(spins, "detach"):  # torch.Tensor support
        spins: np.ndarray = spins.detach().cpu().numpy()  # ty:ignore[call-non-callable]
    else:
        spins = np.asarray(spins)

    if spins.ndim != 2:
        raise ValueError("`spins` must be a 2D array of shape (n_samples, d).")

    unique_vals = np.unique(spins)
    if not np.all(np.isin(unique_vals, [-1, 1])):
        raise ValueError("`spins` must contain only -1 and +1 values.")

    # avoid accidental mutation of caller-provided dict
    log_reg_kwargs = dict(log_reg_kwargs)
    from sklearn.linear_model import LogisticRegression

    d = spins.shape[1]

    h_i_est = np.zeros(d)
    J_ij_est = np.zeros((d, d))
    for i in range(d):
        X = spins[:, np.arange(d) != i]  # features are all spins except i
        y = spins[:, i]  # target is the i-th attribute
        model = LogisticRegression(**log_reg_kwargs)
        model.fit(X, y)
        h_i_est[i] = model.intercept_[0] / 2

        # We only assign the row here to store the estimate of J_{i, j} from regressing i on j.
        J_ij_est[i, np.arange(d) != i] = model.coef_[0] / 2

    # Because regressing i on j gives a slightly different estimate than regressing j on i,
    # we average them at the end to make the interaction matrix J symmetric.
    J_ij_est = (J_ij_est + J_ij_est.T) / 2

    return h_i_est, J_ij_est


def compute_pseudolikelihood(spins, h, J):
    """
    Compute the log-pseudolikelihood of spin configurations under a spin model.
    The pseudolikelihood is the product of conditional probabilities P(s_i | s_{-i})
    for each spin, given all other spins fixed. This is a tractable approximation to
    the full likelihood for undirected graphical models.
    Args:
        spins (np.ndarray): Spin configurations of shape (n_samples, n_attributes).
            Each element should be ±1.
        h (np.ndarray): External magnetic field of shape (n_attributes,).
        J (np.ndarray): Pairwise interaction matrix of shape (n_attributes, n_attributes).
            Should have zeros on the diagonal to exclude self-interactions.
    Returns:
        np.ndarray: Log-pseudolikelihood for each configuration of shape (n_samples,).
            Sum of log conditional probabilities log P(s_i | s_{-i}) across all attributes.
    """

    # Compute local field H_i = h_i + sum_j J_ij s_j
    # Note: J_ij has zeros on the diagonal, so self-interaction is automatically excluded.
    H = h + spins @ J

    # log P(s_i | s_{-i}) = -log(1 + exp(-2 * s_i * H_i))
    # logaddexp(0, x) safely computes log(1 + exp(x)) to prevent overflow
    log_p = -np.logaddexp(0, -2 * spins * H)

    # Sum over all attributes to get the log-pseudolikelihood of each configuration
    return log_p.sum(axis=1)
