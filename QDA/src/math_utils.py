import numpy as np

def compute_class_params(X, y, n_classes):
    """
    Calcola le medie e i priors per ogni classe.
    Ritorna:
    - means: lista di array numpy (uno per classe)
    - priors: lista di float (uno per classe)
    """
    means = []
    priors = []
    n_samples = len(y)
    for k in range(n_classes):
        X_k = X[y == k]
        means.append(np.mean(X_k, axis=0))
        priors.append(len(X_k) / n_samples)
    return means, priors


def compute_class_covariances(X, y, n_classes):
    """
    TASK:
    Invece di una sola matrice media, restituisci una LISTA di matrici.
    Ogni elemento della lista è la covarianza np.cov(X_k) della singola classe.
    """
    covariances = []
    for k in range(n_classes): 
        X_k = X[y == k]
        cov_k = np.cov(X_k, rowvar=False)
        covariances.append(cov_k)
    return covariances


def qda_scoring_function(x, mean_k, inv_cov_k, log_det_k, prior_k):
    """
    TASK:
    Implementa la formula quadratica:
    δ_k(x) = -0.5 * log|Σ_k| - 0.5 * (x - μ_k)ᵀ Σ_k⁻¹ (x - μ_k) + log(π_k)
    
    HINT: (x - μ_k)ᵀ Σ_k⁻¹ (x - μ_k) si può calcolare come:
    diff = x - mean_k
    quad_term = np.sum((diff @ inv_cov_k) * diff, axis=1) # Se x è una matrice
    """
    diff = x - mean_k
    quad_term = np.sum((diff @ inv_cov_k) * diff, axis=1)
    return -0.5 * log_det_k - 0.5 * quad_term + np.log(prior_k)