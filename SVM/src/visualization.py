import matplotlib.pyplot as plt
import numpy as np
from src.model import predict_kernel_svm

def visualize_svm_kernel(X, y_binary, model, title="SVM Decision Boundary", labels=None, house_id=None):
    """
    Visualizza un singolo classificatore binario (1-vs-All) per una casa specifica.
    """
    plt.figure(figsize=(10, 8))
    
    # Colori: Positivo (casa scelta) vs Negativo (altri)
    house_colors = {
        0: '#e63946', # Rosso Grifondoro
        1: '#2a9d8f', # Verde Serpeverde
        2: '#ffb703'  # Giallo Tassorosso
    }
    
    color_main = house_colors.get(house_id, '#1d3557')
    
    # 1. Griglia di Decisione
    ax = plt.gca()
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    xx = np.linspace(x_min, x_max, 100)
    yy = np.linspace(y_min, y_max, 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # Previsioni sulla griglia
    Z_scores = predict_kernel_svm(xy, model)
    Z = Z_scores.reshape(XX.shape)
    
    # Sfondo (Decision Regions)
    ax.contourf(XX, YY, Z > 0, alpha=0.1, colors=[ '#ced4da', color_main])
    
    # Confine di decisione (f(x)=0) e Margini (f(x)=+/-1)
    ax.contour(XX, YY, Z, levels=[-1, 0, 1], colors=[color_main], 
               alpha=0.6, linestyles=[':', '-', ':'], linewidths=[1, 2, 1])
    
    # 2. Plot dei punti
    # Classe Positiva (1)
    plt.scatter(X[y_binary == 1][:, 0], X[y_binary == 1][:, 1], 
                color=color_main, marker='o', label=labels[house_id] if labels else "Target", 
                edgecolors='white', s=70, zorder=10)
    
    # Classe Negativa (-1)
    plt.scatter(X[y_binary == -1][:, 0], X[y_binary == -1][:, 1], 
                color='gray', marker='x', label="Altri", 
                alpha=0.4, s=50, zorder=5)
    
    # Support Vectors
    sv_X = model['sv_X']
    plt.scatter(sv_X[:, 0], sv_X[:, 1], s=150, facecolors='none', 
                edgecolors=color_main, alpha=0.3, label="Support Vectors", zorder=1)
    
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("Coraggio (Normalizzato)")
    plt.ylabel("Ambizione (Normalizzato)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()