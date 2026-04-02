import numpy as np
from data.dataset import get_prepared_data, train_test_split
from src.model import train_svm_kernel_pegasos, predict_kernel_svm
from src.visualization import visualize_svm_kernel
import src.math_utils as mu

def main():
    X, y, nomi, nomi_case = get_prepared_data()
    (X_train, y_train, _), (X_test, y_test, names_test) = train_test_split(X, y, nomi)
    
    X_train_feat, X_test_feat = X_train[:, 1:], X_test[:, 1:]
    
    print("\n--- Scelta del Kernel ---")
    print("1. Lineare\n2. Polinomiale\n3. RBF (Gaussiano)")
    choice = input("Scegli un numero (default 1): ").strip() or "1"
    
    kernels = {
        "1": (mu.linear_kernel, "Linear"),
        "2": (lambda x1, x2: mu.polynomial_kernel(x1, x2, c=1, d=2), "Polynomial"),
        "3": (lambda x1, x2: mu.rbf_kernel(x1, x2, gamma=10), "RBF")
    }
    
    kernel_fn, kernel_name = kernels.get(choice, kernels["1"])
    lambda_param, n_epochs = 0.01, 10
    unique_houses = np.unique(y_train)
    
    print("\n" + "="*40)
    print(f"{'CASA':<15} | {'ACCURATEZZA':<12}")
    print("-" * 40)
    
    results = {}
    for house_id in unique_houses:
        house_name = nomi_case[house_id]
        y_binary_train = np.where(y_train == house_id, 1, -1)
        model = train_svm_kernel_pegasos(X_train_feat, y_binary_train, lambda_param, n_epochs, kernel_fn)
        
        # Valutazione
        y_test_binary = np.where(y_test == house_id, 1, -1)
        scores_test = predict_kernel_svm(X_test_feat, model)
        preds_test = np.sign(scores_test)
        accuracy = np.mean(preds_test == y_test_binary)
        
        results[house_name] = accuracy
        print(f"{house_name:<15} | {accuracy * 100:>10.2f}%")
        
        # Visualizzazione per ogni casa
        visualize_svm_kernel(X_test_feat, y_test_binary, model, 
                             title=f"SVM {kernel_name}: {house_name} vs Tutti",
                             labels=nomi_case, house_id=house_id)
    print("="*40)

if __name__ == "__main__":
    main()