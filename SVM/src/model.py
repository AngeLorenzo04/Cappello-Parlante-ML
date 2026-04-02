import numpy as np

def train_svm_kernel_pegasos(X, y, lambda_param, n_epochs, kernel_fn):
    """
    Implementazione del Kernelized PEGASOS [cite: 151, 152]
    X: Training data (N x D)
    y: Labels {-1, 1}
    kernel_fn: K(x1, x2)
    """
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples) # Coefficienti duali
    
    t = 0
    for epoch in range(1, n_epochs + 1):
        # Mischiamo casualmente per ogni epoca (Stochastic) [cite: 153]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for i in indices:
            t += 1
            # Calcolo f(xi) usando i coefficienti correnti [cite: 161]
            # f(xi) = sum_j alpha_j * y_j * K(xj, xi) / (lambda * t)
            # Nota: Pegasos originale usa alpha come contatore di violazioni
            
            # Identifichiamo i punti che hanno alpha > 0 (Support Vectors)
            sv_indices = np.where(alpha > 0)[0]
            
            if len(sv_indices) == 0:
                score = 0
            else:
                # Vettore dei kernel K(sv, xi)
                k_val = np.array([kernel_fn(X[idx], X[i]) for idx in sv_indices])
                score = np.sum(alpha[sv_indices] * y[sv_indices] * k_val) / (lambda_param * t)
            
            # Condizione di aggiornamento: y_i * f(x_i) < 1
            if y[i] * score < 1:
                alpha[i] += 1
                
    # Restituiamo solo i support vectors per efficienza nell'inferenza [cite: 162]
    sv_idx = np.where(alpha > 0)[0]
    return {
        'sv_X': X[sv_idx],
        'sv_y': y[sv_idx],
        'alpha': alpha[sv_idx],
        'lambda': lambda_param,
        'T': t, # Numero totale di iterazioni
        'kernel_fn': kernel_fn
    }

def predict_kernel_svm(X_test, model):
    """
    Prevede le classi usando il modello kernelizzato.
    f(u) = 1/(lambda*T) * sum_i alpha_i * y_i * K(xi, u)
    """
    sv_X = model['sv_X']
    sv_y = model['sv_y']
    alpha = model['alpha']
    lam = model['lambda']
    T = model['T']
    k_fn = model['kernel_fn']
    
    predictions = []
    for x_u in X_test:
        # Calcolo lo score per ogni punto di test
        k_vals = np.array([k_fn(sv, x_u) for sv in sv_X])
        score = np.sum(alpha * sv_y * k_vals) / (lam * T)
        predictions.append(score)
        
    return np.array(predictions)