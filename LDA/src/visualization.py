import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from src.math_utils import lda_scoring_function

def visualize_lda_boundaries(X, y, names, lda_params, nomi_case, top_n=10):
    """
    Genera una visualizzazione ottimizzata dei territori lineari LDA.
    """
    print("\n📊 Generazione della mappa dei territori di Hogwarts in corso...")
    
    # 1. Configurazione dei Colori (Palette Hogwarts ottimizzata)
    # Colori scuri per i punti: Rosso Grifondoro, Verde Serpeverde, Oro Tassorosso
    colors_points = ['#740001', '#1A472A', '#EEBA30']
    # Colori pastello chiarissimi per lo sfondo: 
    colors_back = ['#FFCCCC', '#CCFFCC', '#FFFFCC'] # Rosa, Verde, Giallo chiarissimi
    cmap_background = ListedColormap(colors_back)

    # 2. Creazione della griglia fitta per i territori
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    # Una griglia molto fitta (0.002) rende i confini retti e fluidi
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.002),
                         np.arange(y_min, y_max, 0.002))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 3. Calcolo dei territori (Niente "Limbo" bianco!)
    all_scores = []
    # Calcoliamo contemporaneamente lo score per tutte le classi su tutti i punti
    for k in range(lda_params["n_classes"]):
        s_k = lda_scoring_function(grid_points, lda_params["means"][k], 
                                    lda_params["inv_cov"], lda_params["priors"][k])
        all_scores.append(s_k)
    
    # Argmax decide chi vince: Z conterrà 0, 1 o 2 per ogni pixel
    Z = np.argmax(np.array(all_scores), axis=0)
    Z = Z.reshape(xx.shape)

    # 4. Disegno del Grafico
    fig, ax = plt.subplots(figsize=(11, 8.5)) # Formato standard A4 per Notion
    
    # Coloriamo lo sfondo (pcolormesh è perfetto per territori netti)
    ax.pcolormesh(xx, yy, Z, cmap=cmap_background, shading='auto', alpha=0.6)

    # Disegniamo i punti di tutti gli studenti (piccoli e sbiaditi)
    for casa_id in range(lda_params["n_classes"]):
        mask = (y == casa_id)
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_points[casa_id], 
                    label=nomi_case[casa_id], edgecolors='k', s=30, alpha=0.5)

    # Evidenziamo i primi 10 con nomi e punti grandi
    for i in range(min(top_n, len(names))):
        ax.scatter(X[i, 0], X[i, 1], c=colors_points[y[i]], 
                    edgecolors='white', s=120, linewidth=2, zorder=5)
        ax.text(X[i, 0] + 0.012, X[i, 1] + 0.012, names[i], 
                fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # Decorazioni Ottimali
    ax.set_xlabel('Coraggio (Normalizzato: 0-1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ambizione (Normalizzato: 0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Hogwarts Territory Map: LDA Decision Regions\n(Distribuzione Coraggio vs Ambizione)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, shadow=True, title="Casate")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()