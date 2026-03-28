from data.dataset import get_prepared_data, train_test_split
from src.model import train_ovr, get_probabilities
from src.visualization import visualize_ovr_boundaries
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. SETUP E TRAINING ---
X_aug, y, nomi, nomi_case = get_prepared_data()
(X_train, y_train, n_train), (X_test, y_test, n_test) = train_test_split(X_aug, y, nomi)

# Otteniamo i pesi (i 3 modelli)
weights = train_ovr(X_train, y_train, n_classes=3, lr=0.5, epochs=2000)

# --- 2. VALUTAZIONE GENERALE ---
probs_test = get_probabilities(X_test, weights)
predizioni_test = np.argmax(probs_test, axis=1)
accuracy = np.mean(predizioni_test == y_test) * 100

print(f"\n✅ Addestramento completato! Accuratezza sul Test Set: {accuracy:.2f}%")
print("-" * 30)

def clear_screen():
    # Pulisce il terminale (funziona su Windows, Mac e Linux)
    os.system('cls' if os.name == 'nt' else 'clear')

# --- DOPO IL TRAINING ---

# --- IL BIVIO PER IL GRAFICO ---
print("\n" + "🪄".center(40, "-"))
scelta_grafico = input("📈 Vuoi visualizzare la mappa dei confini magici? (s/n): ").lower()

if scelta_grafico == 's':
    visualize_ovr_boundaries(X_aug, y, nomi, weights, nomi_case, top_n=10)
else:
    print("Salto la visione profetica.")

house_icons = ["🦁", "🐍", "🦡"]

while True:
    clear_screen()
    print("+" + "-"*48 + "+")
    print("|" + " ".center(48) + "|")
    print("|" + "🧙‍♂️  CERIMONIA DEL CAPPELLO PARLANTE (LR)  🧙‍♂️".center(46) + "|")
    print("|" + " ".center(48) + "|")
    print("+" + "-"*48 + "+")
    
    print("\n📜 Studenti in attesa di smistamento:")
    for i in range(15): 
        print(f"  [{i:2}] {nomi[i]:<20}", end="\t" if i%2!=0 else "\n")
    print("  ... (molti altri)")
    
    scelta = input("\n\n✨ Inserisci l'ID dello studente (o 'q' per uscire): ")
    
    if scelta.lower() == 'q':
        print("\nLa cerimonia è terminata. Buona cena nella Sala Grande!")
        break
    
    try:
        idx = int(scelta)
        studente_X = X_aug[idx:idx+1]
        
        # Calcolo Probabilità
        p = get_probabilities(studente_X, weights)[0]
        casa_id = np.argmax(p)
        casa_predetta = nomi_case[casa_id]
        icon = house_icons[casa_id]
        
        # --- SCHERMATA DEL RISULTATO ---
        clear_screen()
        print("\n" + " ✨ ".center(50, "="))
        print(f"\nIl Cappello viene posto sulla testa di: {nomi[idx].upper()}")
        print("\n\"Mmm... vedo... vedo tutto...\"")
        print(f"\"Vedo una lotta tra {p[0]*100:.1f}% di Coraggio e {p[1]*100:.1f}% di Ambizione...\"")
        
        print("\n" + "+" + "-"*44 + "+")
        print(f"| {icon} RISULTATO: {casa_predetta.upper() + '!':<32} |")
        print("+" + "-"*44 + "+")
        
        print(f"\n📊 Visione interiore del Cappello:")
        for i, casa in enumerate(nomi_case):
            bar = "█" * int(p[i] * 20)
            print(f"  {house_icons[i]} {casa:<10}: {p[i]*100:>5.1f}% {bar}")
            
        print(f"\n(Realtà: {nomi_case[y[idx]]} {house_icons[y[idx]]})")
        
        print("\n" + "="*50)
        input("\nPremere [INVIO] per il prossimo studente...")
        
    except (ValueError, IndexError):
        input("\n❌ ID non trovato tra gli iscritti a Hogwarts! Premi INVIO...")