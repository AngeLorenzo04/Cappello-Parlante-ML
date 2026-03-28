import os
import numpy as np
from data.dataset import get_prepared_data, train_test_split
from src.model import train_lda, predict_lda
from src.visualization import visualize_lda_boundaries

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # --- 1. PREPARAZIONE DATI ---
    # Prendiamo X_aug (che ha il bias), ma per LDA lo togliamo subito
    X_aug, y, nomi, nomi_case = get_prepared_data()
    X_raw = X_aug[:, 1:] # Solo Coraggio e Ambizione
    
    (X_train, y_train, n_train), (X_test, y_test, n_test) = train_test_split(X_raw, y, nomi)
    print("\n📋 Elenco ordinato del Test Set:")
    for i, nome in enumerate(n_test):
        print(f"Riga {i}: {nome}")

    # --- 2. TRAINING (Istantaneo!) ---
    print("🪄 Il Cappello sta analizzando le medie statistiche delle case...")
    lda_params = train_lda(X_train, y_train, n_classes=3)
    
    # --- 3. VALUTAZIONE ---
    y_pred = predict_lda(X_test, lda_params)

    accuracy = np.mean(y_pred == y_test) * 100
    print(f"✅ Analisi completata! Accuratezza LDA: {accuracy:.2f}%")

    # --- 4. GRAFICO ---
    scelta_g = input("\n📈 Vuoi vedere la mappa dei territori LDA? (s/n): ")
    if scelta_g.lower() == 's':
        visualize_lda_boundaries(X_raw, y, nomi, lda_params, nomi_case)

    # --- 5. CERIMONIA ---
    while True:
        clear_screen()
        print("="*45)
        print("🏛️  CERIMONIA LDA: L'ANALISI STATISTICA  🏛️")
        print("="*45)
        
        print("\nStudenti pronti:")
        for i in range(len(nomi)): print(f"[{i}] {nomi[i]}", end="\t" if i%2!=0 else "\n")
        
        scelta = input("\n\nInserisci l'ID (o 'q' per uscire): ")
        if scelta.lower() == 'q': break
        
        try:
            idx = int(scelta)
            studente_X = X_raw[idx:idx+1]
            
            # Predizione singola
            pred_id = predict_lda(studente_X, lda_params)[0]
            
            clear_screen()
            print(f"\n✨ Analisi del DNA magico di: {nomi[idx]} ✨")
            print(f"\nIl Cappello calcola i baricentri delle case...")
            print(f"Risultato: {nomi_case[pred_id].upper()}!")
            print(f"\n(Casa Reale: {nomi_case[y[idx]]})")
            
            print("\n" + "-"*45)
            input("Premere [INVIO] per continuare...")
            
        except:
            input("❌ Errore! Premi INVIO...")

if __name__ == "__main__":
    main()