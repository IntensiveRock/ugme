import numpy as np

def implied_timescales(T, tau):
    """
    Get implied timescales of the TPMs
    """
    ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    ITS = -tau / np.log(ev[1:])
    return ITS

def symmetrize(C):
    """
    Symmetrize the count matrix to enforce detailed balance
    """
    return (C + C.T) / 2.0

def tpm_from_counts(C):
    """
    Compute the transition proability matrices (TPMs) by
    column-normalizing the (symmetrized) count matrices
    """
    return C / C.sum(axis=0, keepdims=True)

def build_metastable_counts(well_depths, connections):
    diag, labels = [], []

    # build diags and assign wells to a state
    for idx, depths in enumerate(well_depths):
        diag.extend(depths)
        labels.extend([idx] * len(depths))

    N = len(diag)
    C = np.zeros((N, N))
    np.fill_diagonal(C, diag)

    # connect neighboring states
    for j in range(N - 1):
        C[j, j + 1] = connections[j]
        C[j + 1, j] = connections[j]

    return C, np.asarray(labels, int)

def assignment_matrix(labels, n_macro=None):
    """
    Build the assignment matrix for the HS projector
    """
    labels = np.asarray(labels, int) # make sure int array
    N = labels.size
    n = (labels.max()+1) if n_macro is None else n_macro
    A = np.zeros((N, n))
    A[np.arange(N), labels] = 1.0

    return A

def hummer_szabo_projector(assnt, pi):
    """
    Computes the Hummer-Szabo (HS) aggregation of the many-state TPMs
    See Hummer & Szabo: J. Phys. Chem. B (2015) 119 (29): 9029–9037.
    """
    PI = np.diag(pi)
    macro = assnt.T @ PI @ assnt
    return PI @ assnt @ np.linalg.inv(macro)

def coarse_grain(T_micro, assnt, PAD):
    """
    Lump the states together with the projector
    """
    return assnt.T @ T_micro @ PAD

def coarse_grained_dynamics(T, assnt, PAD, tsteps, axis="Tnn"):
    """Return stack of macrostate TPMs. axis='Tnn' -> (T,n,n); 'nnT' -> (n,n,T)."""
    n = assnt.shape[1]
    dynamics = np.zeros((tsteps, n, n))

    Tn = np.eye(T.shape[0])
    for k in range(tsteps):
        dynamics[k] = coarse_grain(Tn, assnt, PAD)
        Tn = Tn @ T # propagate the many state and then CG next time through loop

    return dynamics

def equilibrium_populations(C):
    """
    Returns the equilibrium populations
    """
    s = C.sum(axis=0)
    return s / s.sum()



