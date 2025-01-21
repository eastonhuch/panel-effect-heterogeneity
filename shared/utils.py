import numpy as np

def chol_inv_matrix(x):
    chol = np.linalg.cholesky(x)
    chol_inv = np.linalg.solve(chol, np.eye(x.shape[0]))
    inv = chol_inv.T @ chol_inv
    return inv

def chol_inv_stack(x):
    chols = np.linalg.cholesky(x)
    identities = np.zeros_like(x)
    identities[:] = np.eye(x.shape[1])
    chol_invs = np.linalg.solve(chols, identities)
    invs = np.swapaxes(chol_invs, 1, 2) @ chol_invs
    return invs

def float2str_format(x: np.ndarray, digits=3) -> np.ndarray:
    fmt = f"%.{digits}f"
    x_str = np.char.mod(fmt, x)
    return x_str

def print_se_parentheses(x: np.ndarray, x_ses: np.ndarray, digits=3) -> np.ndarray:
    x_str = float2str_format(x, digits)
    x_se_str = float2str_format(x_ses, digits)
    x_se_parentheses = np.char.add(x_str, np.char.add(" (", np.char.add(x_se_str, ")")))
    return x_se_parentheses