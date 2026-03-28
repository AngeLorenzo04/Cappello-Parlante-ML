import numpy as np

def sigmoid(z):
    """Schiaccia i valori tra 0 e 1."""
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    """Calcola quanto il Cappello sta sbagliando (Cross-Entropy)."""
    m = len(y_true)
    loss = (-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)).mean()
    return loss

def compute_gradient(X, y_true, y_pred):
    """Calcola la direzione in cui muovere i pesi."""
    n_samples = X.shape[0]
    error = y_pred - y_true
    gradient = (1/n_samples) * X.T.dot(error)
    return gradient