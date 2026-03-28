import os
import numpy as np
from data.dataset import get_prepared_data, train_test_split
from src.model import train_qda, predict_qda
from src.visualization import visualize_qda_boundaries

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # --- 1. DATI ---
    X_aug, y, nomi, nomi_case = get_prepared_data()
    X_raw = X_aug[:, 1:] # QDA non vuole la colonna di bias (1)
    
    (X_train, y_train, n_train), (X_test, y_test, n_test) = train_test_split(X_raw, y, nomi)

    # --- 2. TRAINING QDA ---
    print("🔮 Il Cappello sta analizzando le forme uniche di ogni casa...")
    qda_params = train_qda(X_train, y_train, n_classes=3)
    
    # --- 3. ACCURATEZZA ---
    y_pred = predict_qda(X_test, qda_params)
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"✅ QDA Addestrata! Accuratezza: {accuracy:.2f}%")

    # --- 4. GRAFICO ---
    if input("\n📈 Vuoi vedere la mappa curva della QDA? (s/n): ").lower() == 's':
        visualize_qda_boundaries(X_raw, y, nomi, qda_params, nomi_case)

    # --- 5. CERIMONIA ---
    while True:
        clear_screen()
        print("="*45)
        print("🪄  CERIMONIA QDA: IL CAPPELLO SI EVOLVE  🪄")
        print("="*45)
        
        for i in range(10): print(f"[{i}] {nomi[i]}", end="\t" if i%2!=0 else "\n")
        
        scelta = input("\n\nID studente (o 'q'): ")
        if scelta.lower() == 'q': break
        
        try:
            idx = int(scelta)
            pred_id = predict_qda(X_raw[idx:idx+1], qda_params)[0]
            
            clear_screen()
            print(f"✨ Smistamento QDA per: {nomi[idx]} ✨")
            print(f"\nIl Cappello sente la forma della tua anima...")
            print(f"Risultato: {nomi_case[pred_id].upper()}!")
            print(f"\n(Realtà: {nomi_case[y[idx]]})")
            input("\n[INVIO] per continuare...")
        except:
            input("❌ Errore. Premi INVIO...")

if __name__ == "__main__":
    main()