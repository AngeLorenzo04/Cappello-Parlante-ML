# Quadratic Discriminant Analysis (QDA) Documentation

Quadratic Discriminant Analysis is a classifier that models the probability density of each class using a Gaussian distribution. Unlike LDA, it assumes that each class has its **own covariance matrix**.

## 1. Mathematical Foundation

The goal of QDA is to allow the decision boundaries to be non-linear (quadratic).

### The Scoring Function
For a given observation $x$, the score for class $k$ is calculated as:
$$\delta_k(x) = -\frac{1}{2} \log |\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log \pi_k$$

Where:
- $\mu_k$: Mean vector for class $k$.
- $\Sigma_k$: Individual covariance matrix for class $k$.
- $\pi_k$: Prior probability of class $k$.
- $|\Sigma_k|$: The determinant of the covariance matrix.
- $\Sigma_k^{-1}$: The inverse of the covariance matrix.

The predicted class is the one that maximizes $\delta_k(x)$.

## 2. Key Assumption
The critical assumption of QDA is that **$\Sigma_k \neq \Sigma_{j}$ for $k \ne j$**. This means the classes have different shapes and orientations in feature space, which leads to **quadratic decision boundaries**.

## 3. Implementation Details

In this project, the implementation is divided as follows:

### `src/math_utils.py`
- `compute_class_params`: Calculates $\mu_k$ and $\pi_k$.
- `compute_class_covariances`: Calculates the individual covariance matrix $\Sigma_k$ for each house.
- `qda_scoring_function`: Implements the formula for $\delta_k(x)$.

### `src/model.py`
- `train_qda`: Orchestrates the calculation of $\mu_k$, $\pi_k$, $\Sigma_k^{-1}$, and $\log |\Sigma_k|$.
- `predict_qda`: Iterates through classes using house-specific parameters.

## 4. Advantages
- **Flexibility**: Can model decision boundaries that are curves, circles, or ellipses.
- **Accuracy**: Often outperforms LDA when the covariance matrices are significantly different.
- **Robustness**: Does not require shared variance across classes.

> [!WARNING]
> QDA requires estimating more parameters than LDA ($K$ covariance matrices instead of $1$). Be careful with small datasets where $\Sigma_k$ might become singular.
