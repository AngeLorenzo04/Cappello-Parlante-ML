import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from src.math_utils import qda_scoring_function

def visualize_qda_boundaries(X, y, names, qda_params, nomi_case, top_n=10):
    """Genera la visualizzazione dei territori quadratici (curvi) della QDA."""
    print("\n📊 Generazione della mappa quadratica di Hogwarts...")
    
    # 1. Palette Colori Hogwarts
    colors_points = ['#740001', '#1A472A', '#EEBA30'] # Rosso, Verde, Oro
    colors_back = ['#FFCCCC', '#CCFFCC', '#FFFFCC']  # Rosa, Verde, Giallo (pastello)
    cmap_background = ListedColormap(colors_back)

    # 2. Griglia (Meshgrid)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005),
                         np.arange(y_min, y_max, 0.005))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 3. Calcolo Score QDA (Ogni classe ha i suoi parametri unici)
    all_scores = []
    for k in range(qda_params["n_classes"]):
        s_k = qda_scoring_function(
            grid_points, 
            qda_params["means"][k], 
            qda_params["inv_covs"][k], 
            qda_params["log_dets"][k],
            qda_params["priors"][k]
        )
        all_scores.append(s_k)
    
    # Vincitore per ogni pixel
    Z = np.argmax(np.array(all_scores), axis=0)
    Z = Z.reshape(xx.shape)

    # 4. Disegno
    plt.figure(figsize=(11, 8.5))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_background, shading='auto', alpha=0.6)

    # Punti di tutti gli studenti
    for casa_id in range(qda_params["n_classes"]):
        mask = (y == casa_id)
        plt.scatter(X[mask, 0], X[mask, 1], c=colors_points[casa_id], 
                    label=nomi_case[casa_id], edgecolors='k', s=30, alpha=0.4)

    # Primi 10 con nomi
    for i in range(min(top_n, len(names))):
        plt.scatter(X[i, 0], X[i, 1], c=colors_points[y[i]], edgecolors='white', s=120, zorder=5)
        plt.text(X[i, 0] + 0.012, X[i, 1] + 0.012, names[i], fontsize=9, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title('Hogwarts Quadratic Map: QDA Decision Regions\n(I confini si curvano!)', fontsize=14, fontweight='bold')
    plt.xlabel('Coraggio', fontweight='bold')
    plt.ylabel('Ambizione', fontweight='bold')
    plt.legend(title="Case")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()