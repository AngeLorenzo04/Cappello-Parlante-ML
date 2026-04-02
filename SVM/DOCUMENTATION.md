# Support Vector Machines (SVM) Documentation

Support Vector Machines (SVM) are discriminative classifiers that aim to find the **Maximum Margin Hyperplane** separating different classes.

## 1. Mathematical Foundation

The core goal of SVM is to find a decision hyperplane $f(x) = w^T x + b = 0$ that correctly separates the data points while maximizing the distance (margin) to the nearest points.

### The Objective Function
The primal optimization problem is:
$$\min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^m \max(0, 1 - y_i(w^T x_i + b))$$

Where:
- $w$: Normal vector to the hyperplane.
- $C$: Penalty parameter for misclassification (controls the bias/variance trade-off).
- $\xi = \max(0, 1 - y_i f(x_i))$: The **Slack variables** that allow some points to be within the margin or misclassified (Soft Margin).

## 2. Kernel Trick
Kernel functions allow the SVM to find non-linear decision boundaries by implicitly mapping the input features to a higher-dimensional space.

### Available Kernels in this Project
1.  **Linear**: $K(x_i, x_j) = x_i^T x_j$
2.  **Polynomial**: $K(x_i, x_j) = (x_i^T x_j + c)^d$ (Maps data into $d$-degree polynomial space).
3.  **RBF (Gaussian)**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$ (Maps data into infinite-dimensional radial space).

## 3. Optimization: PEGASOS
This project implements the **PEGASOS** (Primal Estimated sub-GrAdient SOlver) algorithm:
- It is a **Stochastic Sub-gradient Descent** method for the primal SVM objective.
- For the kernelized version, it tracks **Support Vectors** (points that violate the margin) and their dual coefficients $\alpha_i$.

## 4. Implementation Details

### `src/math_utils.py`
- `linear_kernel`, `polynomial_kernel`, `rbf_kernel`: Implements the $\phi(x)$ mapping functions.
- `hinge_loss`: Calculates the cost function $\max(0, 1 - y_i f(x_i))$.

### `src/model.py`
- `train_svm_kernel_pegasos`: Performs the iterative update of coefficients $\alpha$ based on classification errors.
- `predict_kernel_svm`: Computes the weighted kernel sum to predict new house assignments.

## 5. Advantages
- **Maximum Margin**: Generally generalizes better than probabilistic models like Logistic Regression.
- **Non-Linearity**: Easily handles complex house boundaries with RBF kernels.
- **Support-Vector Based**: Only a subset of the training points (SVs) are needed for prediction, making it memory-efficient for inference.

> [!IMPORTANT]
> A larger $\lambda$ (regularization) in the Pegasos algorithm results in a **wider margin**, allowing for more slack but reducing training accuracy to prevent overfitting.
