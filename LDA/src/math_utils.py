import numpy as np

def compute_class_params(X, y, n_classes):
    """Calcola medie e prior per ogni classe."""
    means = []
    priors = []
    n_total = len(y)
    for k in range(n_classes):
        X_k = X[y == k]
        means.append(np.mean(X_k, axis=0))
        priors.append(len(X_k) / n_total)
    return np.array(means), np.array(priors)

def compute_shared_covariance(X, y, n_classes):
    """
    Calcola la Pooled Covariance Matrix (Σ).
    Pesa ogni classe per (N_k - 1) per gestire classi sbilanciate.
    """
    n_features = X.shape[1]
    n_total = len(y)
    shared_cov = np.zeros((n_features, n_features))
    
    for k in range(n_classes):
        X_k = X[y == k]
        n_k = len(X_k)
        if n_k > 1:
            # Formula pesata: shared_cov += (N_k - 1) * Σ_k
            shared_cov += (n_k - 1) * np.cov(X_k, rowvar=False)
            
    # Divide per i gradi di libertà totali (N - K)
    return shared_cov / (n_total - n_classes)

def lda_scoring_function(x, mean_k, inv_cov, prior_k):
    """
    Implementa la funzione lineare discriminante (δ_k) usando prodotti matriciali (@).
    Formula: δ_k(x) = x Σ⁻¹ μ_kᵀ - 0.5 μ_k Σ⁻¹ μ_kᵀ + log(π_k)
    """
    # term1: x @ inv_cov @ mu_k (Questa è la proiezione lineare)
    # x può essere una matrice (N_griglia, 2), il risultato sarà un vettore (N_griglia,)
    term1 = x @ inv_cov @ mean_k
    
    # term2: 0.5 * mu_k @ inv_cov @ mu_k (Il "centro" statico della casa)
    term2 = 0.5 * (mean_k @ inv_cov @ mean_k)
    
    # term3: log(prior)
    term3 = np.log(prior_k)
    
    return term1 - term2 + term3