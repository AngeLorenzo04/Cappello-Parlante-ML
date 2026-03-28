import numpy as np
from src.math_utils import compute_class_params, compute_class_covariances, qda_scoring_function

def train_qda(X, y, n_classes):
    """
    TASK:
    1. Ottieni medie e priors.
    2. Ottieni la lista delle covarianze (una per classe).
    3. Per ogni classe, calcola:
       - L'inversa della sua covarianza (np.linalg.inv).
       - Il logaritmo del determinante (np.log(np.linalg.det(...))).
    4. Ritorna tutto in un dizionario.
    """
    means, priors = compute_class_params(X,y,n_classes)
    covariances = compute_class_covariances(X,y,n_classes)
    inv_covs = [np.linalg.inv(cov) for cov in covariances]
    log_dets = [np.log(np.linalg.det(cov)) for cov in covariances]
    return {
        "means": means,
        "priors": priors,
        "inv_covs": inv_covs,
        "log_dets": log_dets,
        "n_classes": n_classes
    }

def predict_qda(X, qda_params):
    """
    TASK:
    Simile alla LDA, ma cicla usando i parametri specifici di ogni classe 
    (ogni classe ha la sua inv_cov e il suo log_det).
    """
    all_scores = []
    
    for k in range(qda_params["n_classes"]):
        # Usa qda_scoring_function passando X e i parametri della classe k
        score_k = qda_scoring_function(
            X, 
            qda_params["means"][k], 
            qda_params["inv_covs"][k], 
            qda_params["log_dets"][k], 
            qda_params["priors"][k]
        )
        all_scores.append(score_k)
    
    # Trasforma in array e usa np.argmax
    all_scores = np.array(all_scores).T # Ora è (N_studenti, 3_classi)
    return np.argmax(all_scores, axis=1)