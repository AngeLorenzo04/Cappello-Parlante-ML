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
    house_icons = ["🦁", "🐍", "🦡"]
    while True:
        clear_screen()
        print("+" + "-"*48 + "+")
        print("|" + " ".center(48) + "|")
        print("|" + "🪄  CERIMONIA DI SMISTAMENTO (QDA)  🪄".center(46) + "|")
        print("|" + " ".center(48) + "|")
        print("+" + "-"*48 + "+")
        
        print("\n📜 Studenti pronti per l'evoluzione:")
        for i in range(15): 
            print(f"  [{i:2}] {nomi[i]:<20}", end="\t" if i%2!=0 else "\n")
        print("  ... (molti altri)")
        
        scelta = input("\n\n✨ ID dello studente da analizzare (o 'q'): ")
        if scelta.lower() == 'q': break
        
        try:
            idx = int(scelta)
            pred_id = predict_qda(X_raw[idx:idx+1], qda_params)[0]
            icon = house_icons[pred_id]
            
            clear_screen()
            print("\n" + " ✨ ".center(50, "="))
            print(f"\nIl Cappello freme sulla testa di: {nomi[idx].upper()}")
            print("\n\"Ah! Una forma complessa... una mente non lineare!\"")
            print("\"Le parabole del tuo destino sono chiare...\"")
            
            print("\n" + "+" + "-"*44 + "+")
            print(f"| {icon} RISULTATO: {nomi_case[pred_id].upper() + '!':<32} |")
            print("+" + "-"*44 + "+")
            
            print(f"\n(Verità Incisa: {nomi_case[y[idx]]} {house_icons[y[idx]]})")
            
            print("\n" + "="*50)
            input("\nPremere [INVIO] per il prossimo studente...")
            
        except (ValueError, IndexError):
            input("\n❌ ID non presente nel Libro degli Studenti! Premi INVIO...")

if __name__ == "__main__":
    main()