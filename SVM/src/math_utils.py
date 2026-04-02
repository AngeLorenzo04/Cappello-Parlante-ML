import numpy as np

# --- KERNEL FUNCTIONS --- [cite: 73, 74]

def linear_kernel(x1, x2):
    """K(xi, xj) = <xi, xj>""" # [cite: 75]
    # Se x2 è una matrice (N x D), restituisce un vettore di prodotti scalari
    return np.dot(x1, x2.T)

def polynomial_kernel(x1, x2, c=1, d=3):
    """K(xi, xj) = (<xi, xj> + c)^d""" # [cite: 76]
    return np.power(np.dot(x1, x2.T) + c, d)

def rbf_kernel(x1, x2, gamma=1.0):
    """K(xi, xj) = exp(-gamma * ||xi - xj||^2)""" # [cite: 76]
    # Se x2 è una matrice, calcoliamo le distanze al quadrato vettorialmente
    if x2.ndim == 2:
        dist_sq = np.sum((x1 - x2)**2, axis=1)
    else:
        dist_sq = np.sum((x1 - x2)**2)
    return np.exp(-gamma * dist_sq)

# --- LOSS & GRADIENT --- [cite: 109, 131, 132]

def hinge_loss(y_true, y_pred):
    """max(0, 1 - y_i * f(x_i))""" # [cite: 106, 112]
    return np.maximum(0, 1 - y_true * y_pred)