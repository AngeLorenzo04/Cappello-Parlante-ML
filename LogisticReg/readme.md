# Logistic Regression (LOGREG) Documentation

Logistic Regression is a probabilistic linear classifier that uses the **Sigmoid function** to map linear results to a range between 0 and 1.

## 1. Mathematical Foundation

The fundamental mapping is given by:
$$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

Where:
- $w$: Vector of weights for features (Courage, Ambition).
- $b$: Bias term (intercept).
- $\sigma(z)$: The sigmoid function.

### Optimization: Gradient Descent
To find the optimal weights $w$ and bias $b$, we minimize the **Binary Cross-Entropy Loss**:
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

The gradients are calculated as:
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$$

## 2. Multiclass Strategy: One-vs-Rest (OvR)
Because pure Logistic Regression is a binary classifier, we use the **One-vs-Rest** strategy to handle the multiple house classes:
1.  Train a separate binary classifier for each house (e.g., Grifondoro vs Others).
2.  During prediction, apply all classifiers to the input.
3.  Assign the house that produces the **highest probability**.

## 3. Implementation Details

In this project, the implementation is divided as follows:

### `src/math_utils.py`
- `sigmoid`: Computes the $\sigma(z)$ function.
- `compute_gradient`: Calculates the gradient vectors for backpropagation.
- `compute_loss`: Computes the Binary Cross-Entropy Loss.

### `src/model.py`
- `train_ovr`: Iterates through each class and runs Gradient Descent for the given number of epochs.
- `predict_ovr`: Calculates probabilities across all models and chooses the max.

## 4. Advantages
- **Probabilistic**: Provides a clear probability score for each class.
- **Interpretable**: The weights $w$ directly show the influence of each feature.
- **Fast Inference**: Prediction only requires a dot product and a sigmoid.

> [!CAUTION]
> Logistic Regression assumes a linear relationship between the log-odds and the features. If your house boundaries are highly non-linear, you may need Feature Engineering or SVM with non-linear Kernels.
