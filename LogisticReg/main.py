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
print("\n" + "="*30)
scelta_grafico = input("📈 Vuoi visualizzare il grafico dei confini di decisione? (s/n): ").lower()

if scelta_grafico == 's':
    # Chiamiamo la funzione dal file separato
    visualize_ovr_boundaries(X_aug, y, nomi, weights, nomi_case, top_n=10)
else:
    print("Salto la visualizzazione grafica.")

while True:
    clear_screen()
    print("="*40)
    print("🧙‍♂️ CERIMONIA DEL CAPPELLO PARLANTE 🧙‍♂️")
    print("="*40)
    
    # Mostriamo solo una manciata di nomi per non intasare lo schermo
    print("\nAlcuni studenti pronti per lo smistamento:")
    for i in range(10): 
        print(f"[{i}] {nomi[i]}")
    print("... (e molti altri)")
    
    scelta = input("\nInserisci l'ID dello studente (o 'q' per uscire): ")
    
    if scelta.lower() == 'q':
        print("\nLa cerimonia è terminata. Buona cena nella Sala Grande!")
        break
    
    try:
        idx = int(scelta)
        studente_X = X_aug[idx:idx+1]
        
        # Calcolo Probabilità
        p = get_probabilities(studente_X, weights)[0]
        casa_predetta = nomi_case[np.argmax(p)]
        
        # --- SCHERMATA DEL RISULTATO ---
        clear_screen()
        print(f"\n✨ Il Cappello è stato posto sulla testa di: {nomi[idx]} ✨")
        print("\nIl Cappello mormora tra sé e sé...")
        print(f"\"Vedo... {p[0]*100:.1f}% di Coraggio, {p[1]*100:.1f}% di Ambizione...\"")
        print("\n" + "!"*10 + f" {casa_predetta.upper()}! " + "!"*10)
        
        print(f"\n📊 Analisi dettagliata:")
        print(f"🦁 Grifondoro: {p[0]*100:.1f}%")
        print(f"🐍 Serpeverde: {p[1]*100:.1f}%")
        print(f"🦡 Tassorosso:  {p[2]*100:.1f}%")
        
        # --- IL PULSANTE DI PAUSA ---
        print("\n" + "-"*40)
        input("Premere [INVIO] per il prossimo studente...")
        
    except (ValueError, IndexError):
        input("\n❌ ID non valido! Premi INVIO per riprovare...")