import numpy as np 
from src.math_utils import compute_class_params, compute_shared_covariance, lda_scoring_function

def train_lda(X, y, n_classes):
    """
    TASK:
    1. Calcola medie e priors usando compute_class_params.
    2. Calcola la covarianza condivisa usando compute_shared_covariance.
    3. Calcola l'inversa della covarianza (np.linalg.inv).
    4. Ritorna un dizionario con i risultati.
    """

    means, priors = compute_class_params(X,y,n_classes)
    shared_cov = compute_shared_covariance(X,y,n_classes)
    inv_cov = np.linalg.inv(shared_cov)

    return {
        "means": means,
        "priors": priors,
        "inv_cov": inv_cov,
        "n_classes": n_classes
    }

def predict_lda(X, lda_params):
    """
    TASK:
    1. Per ogni classe k, calcola lo score per tutti gli studenti in X.
    2. Confronta gli score e prendi il massimo.
    """
    all_scores = []
    
    for k in range(lda_params["n_classes"]):
        # Usa lda_scoring_function passando X e i parametri della classe k
        score_k = lda_scoring_function(
            X, 
            lda_params["means"][k], 
            lda_params["inv_cov"], 
            lda_params["priors"][k]
        )
        all_scores.append(score_k)
    
    # Trasforma in array e usa np.argmax
    all_scores = np.array(all_scores).T # Ora è (N_studenti, 3_classi)
    print("all_scores: ", all_scores)
    return np.argmax(all_scores, axis=1)