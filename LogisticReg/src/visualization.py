import matplotlib.pyplot as plt
import numpy as np

# Aggiungi questa funzione nel tuo main.py
def visualize_ovr_boundaries(X_aug, y, names, all_weights, noms_case, top_n=10):
    # 1. Prepariamo la griglia per disegnare i confini (Decision Boundaries)
    x_min, x_max = X_aug[:, 1].min() - 0.1, X_aug[:, 1].max() + 0.1
    y_min, y_max = X_aug[:, 2].min() - 0.1, X_aug[:, 2].max() + 0.1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Grid_X è la matrice di test [Nx3] per disegnare le probabilità
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Aggiungiamo la colonna di 1 per il bias!
    ones_grid = np.ones((grid_points.shape[0], 1))
    grid_X = np.hstack((ones_grid, grid_points))
    
    # 2. Setup del grafico
    plt.figure(figsize=(10, 8))
    
    # 3. Disegniamo le 3 sigmoide come curve di livello (contorni)
    colors_boundary = ['red', 'green', 'yellow']
    probs_grid = []
    
    for i, w in enumerate(all_weights):
        # Calcoliamo la sigmoide su tutta la griglia
        z = grid_X @ w
        p = 1 / (1 + np.exp(-z)) # La nostra sigmoide
        
        # Disegniamo solo la linea del 50% (la linea di confine)
        probs_grid.append(p)
        contour = plt.contour(xx, yy, p.reshape(xx.shape), levels=[0.5], colors=colors_boundary[i], linewidths=2)
        # Etichettiamo la linea
        # plt.clabel(contour, inline=True, fontsize=10, fmt={0.5: noms_case[i]})
        
    # 4. Disegniamo tutti i punti del dataset
    colors_points = {0: 'red', 1: 'green', 2: 'yellow'}
    for casa_id in range(3):
        mask = (y == casa_id)
        plt.scatter(X_aug[mask, 1], X_aug[mask, 2], c=colors_points[casa_id], label=noms_case[casa_id], edgecolors='k', alpha=0.3)
        
    # 5. Collochiamo i primi 10 esempi (con etichetta e nomi)
    # Mostriamo solo i primi top_n, con opacità massima e nomi
    top_indices = np.arange(top_n)
    
    plt.scatter(X_aug[top_indices, 1], X_aug[top_indices, 2], c=[colors_points[yi] for yi in y[top_indices]], edgecolors='k', s=100)
    
    # Aggiungiamo i nomi come etichette
    for idx in top_indices:
        plt.text(X_aug[idx, 1] + 0.01, X_aug[idx, 2] + 0.01, names[idx], fontsize=9)
        
    # 6. Decorazioni
    plt.xlabel('Coraggio (Normalizzato)')
    plt.ylabel('Ambizione (Normalizzato)')
    plt.title('Sorting Hat: One-vs-Rest Decision Boundaries\n(I primi 10 personaggi)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
