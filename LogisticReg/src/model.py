import numpy as np
from src.math_utils import sigmoid, compute_gradient, compute_loss

def train_ovr(X, y, n_classes, lr=0.1, epochs=1000):
    all_weights = []
    
    for c in range(n_classes):
        print(f"🪄 Addestramento per la casa {c}...")
        X = np.array(X)
        y = np.array(y)
        # 1. Prepariamo le label "1 contro tutti"
        y_binary = (y == c).astype(int)
        
        # 2. Inizializziamo i pesi (Bias, Coraggio, Ambizione)
        w = np.zeros(X.shape[1])
        
        # 3. Gradient Descent
        for epoch in range(epochs):
            # Calcoliamo il punteggio lineare z
            z = X @ w
            # Trasformiamo in probabilità
            y_pred = sigmoid(z)
            
            # Calcoliamo la direzione dell'errore (il gradiente)
            grad = compute_gradient(X, y_binary, y_pred)
            
            # AGGIORNAMENTO: Ci muoviamo contro il gradiente
            w = w - lr * grad
            
            # (Opzionale) Stampa la loss ogni 100 epoche per vedere se scende
            if epoch % 100 == 0:
                loss = compute_loss(y_binary, y_pred)
                print(f"Epoca {epoch}: Loss = {loss:.4f}")
                
        all_weights.append(w)
        
    return all_weights

def predict_ovr(X, all_weights):
    # Calcoliamo le probabilità per ogni casa
    # Risultato: una matrice dove ogni colonna è una casa
    probs = []
    for w in all_weights:
        probs.append(sigmoid(X @ w))
    
    probs = np.array(probs).T # Trasponiamo per avere [studente, prob_casa]
    
    # Scegliamo l'indice della probabilità massima per ogni riga
    return np.argmax(probs, axis=1)

def get_probabilities(X, all_weights):
    """Restituisce una matrice [studenti x 3] con le probabilità per ogni casa."""
    probs = []
    for w in all_weights:
        # Usiamo la sigmoide sui punteggi lineari
        probs.append(sigmoid(X @ w))
    
    # Trasponiamo per avere una riga per studente e una colonna per casa
    return np.array(probs).T