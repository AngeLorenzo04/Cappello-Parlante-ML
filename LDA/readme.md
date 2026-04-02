# Linear Discriminant Analysis (LDA) Documentation

Linear Discriminant Analysis is a classifier that models the probability density of each class using a Gaussian distribution. It assumes that all classes share the **same covariance matrix**.

## 1. Mathematical Foundation

The goal of LDA is to find a linear combination of features that maximizes the separation between classes.

### The Scoring Function
For a given observation $x$, the score for class $k$ is calculated as:
$$\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k$$

Where:
- $\mu_k$: Mean vector of features for class $k$.
- $\Sigma$: Shared covariance matrix across all classes.
- $\pi_k$: Prior probability of class $k$ (percentage of samples belonging to $k$).
- $\Sigma^{-1}$: The inverse of the shared covariance matrix.

The predicted class is the one that maximizes $\delta_k(x)$.

## 2. Key Assumption
The critical assumption of LDA is that **$\Sigma_k = \Sigma$ for all $k$**. This means the classes are assumed to have the same shape and orientation in feature space, which leads to **linear decision boundaries**.

## 3. Implementation Details

In this project, the implementation is divided as follows:

### `src/math_utils.py`
- `compute_class_params`: Calculates $\mu_k$ and $\pi_k$.
- `compute_shared_covariance`: Calculates the pooled covariance matrix $\Sigma$.
- `lda_scoring_function`: Implements the formula for $\delta_k(x)$.

### `src/model.py`
- `train_lda`: Orchestrates the calculation of $\mu_k$, $\pi_k$, and $\Sigma^{-1}$.
- `predict_lda`: Iterates through classes to find the highest score.

## 4. Advantages
- **Robustness**: Works well even if the Gaussian assumption is slightly violated.
- **Simplicity**: Fast to train as it only requires calculating means and one matrix inversion.
- **Dimensionality Reduction**: Can be used to project data into a lower-dimensional space while preserving class separation.

> [!TIP]
> Use LDA when your features are continuous and you have a reason to believe the casa classes have similar variance structures.
