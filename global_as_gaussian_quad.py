import numpy as np
from typing import Callable
from scipy.integrate import newton_cotes
from scipy.special import roots_legendre
import scipy.stats as stats

def g_normal(v: np.ndarray) -> np.ndarray:
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * v**2)

def get_newton_cotes_closed(n: int):
    if n < 1:
        raise ValueError("Order n must be at least 1.")
    elif n == 1:
        weights_integral = np.array([1.0])
    else:
        weights_integral, _ = newton_cotes(n-1)
    weights_average = weights_integral / sum(weights_integral)
    points = np.linspace(0, 1, n + 2)
    
    return points[1:-1], weights_average

def get_gauss_legendre(n: int):
    a = 0.0
    b = 1.0
    nodes_std, weights_std = roots_legendre(n)
    nodes_transformed = 0.5 * (b - a) * nodes_std + 0.5 * (a + b)
    weights_transformed = 0.5 * (b - a) * weights_std
    return nodes_transformed, weights_transformed

def calculate_finite_difference_matrix(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_points: int = 2
) -> np.ndarray:

    sample_len, dim = X.shape
    fd_matrix = np.zeros_like(X, dtype=float)
    f_value_old = f(X)

    points, weights = get_gauss_legendre(n_points)
    1+1
    for i in range(sample_len):
        for j in range(dim):            
            eval_batch_left = np.tile(X[i, :], (n_points, 1))            
            eval_batch_left[:, j] *= points
            f_values_left = f(eval_batch_left)
            x_left = (eval_batch_left[:, j] - np.tile(X[i, j], (n_points, 1))[0])
            y_left = (f_values_left - f_value_old[i])
            average_value_left = np.dot(y_left / x_left, weights)

            eval_batch_right = np.tile(X[i, :], (n_points, 1))            
            eval_batch_right[:, j] = eval_batch_right[:, j] + points * (1 - eval_batch_right[:, j])
            f_values_right = f(eval_batch_right)
            x_right = (eval_batch_right[:, j] - np.tile(X[i, j], (n_points, 1))[0])
            y_right = (f_values_right - f_value_old[i])
            average_value_right = np.dot(y_right / x_right, weights)
            
            fd_matrix[i, j] = average_value_left * X[i, j] + average_value_right * (1-X[i, j])
            #print(j, eval_batch_left, eval_batch_right)            
    return fd_matrix



# def calculate_finite_difference_matrix_normal1(
#     f: Callable[[np.ndarray], np.ndarray],
#     X: np.ndarray,
#     n_points: int = 2
# ) -> np.ndarray:

#     sample_len, dim = X.shape
#     fd_matrix = np.zeros_like(X, dtype=float)
#     f_value_old = f(X)

#     points, weights = get_gauss_legendre(n_points)
#     1+1
#     for i in range(sample_len):
#         for j in range(dim):            
#             eval_batch_left = np.tile(X[i, :], (n_points, 1))            
#             eval_batch_left[:, j] -= points / (1 - points)
#             f_values_left = f(eval_batch_left)
#             x_left = -points / (1 - points)
#             y_left = (f_values_left - f_value_old[i])
#             weights_left = -weights / (1 - points) ** 2 * stats.norm.pdf(eval_batch_left[:, j])
#             weights_left /= np.sum(weights_left)
#             average_value_left = np.dot(y_left / x_left, weights_left)

#             eval_batch_right = np.tile(X[i, :], (n_points, 1))            
#             eval_batch_right[:, j] += points / (1 - points)
#             f_values_right = f(eval_batch_right)
#             x_right = points / (1 - points)
#             y_right = (f_values_right - f_value_old[i])
#             weights_right = weights / (1 - points) ** 2 * stats.norm.pdf(eval_batch_right[:, j])
#             weights_right /= np.sum(weights_right)
#             average_value_right = np.dot(y_right / x_right, weights_right)
            
#             fd_matrix[i, j] = average_value_left * stats.norm.cdf(X[i, j]) + average_value_right * (1-stats.norm.cdf(X[i, j]))
#             #print(j, eval_batch_left, eval_batch_right)            
#     return fd_matrix
import numpy as np
from scipy import stats
from scipy.special import roots_legendre
from typing import Callable

def calculate_finite_difference_matrix_normal1(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_points: int = 2
) -> np.ndarray:
    sample_len, dim = X.shape
    f_value_old = f(X)
    
    nodes_std, weights_std = roots_legendre(n_points)
    points = 0.5 * nodes_std + 0.5
    weights = 0.5 * weights_std
    delta = points / (1 - points)
    
    total_perturbations = sample_len * dim * n_points
    base_X = np.repeat(X, dim * n_points, axis=0)
    
    perturbation_matrix = np.zeros_like(base_X)
    delta_values = np.tile(np.tile(-delta, dim), sample_len)
    row_indices = np.arange(total_perturbations)
    col_indices = np.tile(np.repeat(np.arange(dim), n_points), sample_len)
    perturbation_matrix[row_indices, col_indices] = delta_values
    
    eval_batch_left = base_X + perturbation_matrix
    eval_batch_right = base_X - perturbation_matrix
    eval_batch_combined = np.vstack([eval_batch_left, eval_batch_right])
    
    f_values_flat = f(eval_batch_combined)
    f_values_left = f_values_flat[:total_perturbations].reshape(sample_len, dim, n_points)
    f_values_right = f_values_flat[total_perturbations:].reshape(sample_len, dim, n_points)
    
    y_left = f_values_left - f_value_old[:, np.newaxis, np.newaxis]
    ratio_left = np.divide(y_left, -delta, out=np.zeros_like(y_left), where=delta != 0)
    perturbed_coords = eval_batch_left[row_indices, col_indices].reshape(sample_len, dim, n_points)
    weights_left = -weights / (1 - points)**2 * stats.norm.pdf(perturbed_coords)
    weights_left /= np.sum(weights_left, axis=2, keepdims=True)
    average_value_left = np.sum(ratio_left * weights_left, axis=2)
    
    y_right = f_values_right - f_value_old[:, np.newaxis, np.newaxis]
    ratio_right = np.divide(y_right, delta, out=np.zeros_like(y_right), where=delta != 0)
    perturbed_coords = eval_batch_right[row_indices, col_indices].reshape(sample_len, dim, n_points)
    weights_right = weights / (1 - points)**2 * stats.norm.pdf(perturbed_coords)
    weights_right /= np.sum(weights_right, axis=2, keepdims=True)
    average_value_right = np.sum(ratio_right * weights_right, axis=2)
    
    cdf_X = stats.norm.cdf(X)
    fd_matrix = average_value_left * cdf_X + average_value_right * (1 - cdf_X)
    
    return fd_matrix


import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import stats
from typing import Callable

def calculate_sensitivity_matrix(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_points: int = 8  # Higher n_points for better accuracy
) -> np.ndarray:
    """
    Calculates a sensitivity/covariance-like matrix C for a function f
    at points X using Gaussian-Legendre quadrature.

    Args:
        f: The function to analyze, mapping (n, dim) -> (n,).
        X: The points at which to calculate the matrix, shape (sample_len, dim).
        n_points: The number of quadrature points to use for the approximation.

    Returns:
        A matrix C of shape (sample_len, dim, dim).
    """
    sample_len, dim = X.shape
    f_value_old = f(X)

    # --- Quadrature Setup (Identical to the previous function) ---
    nodes_std, weights_std = leggauss(n_points)
    points = 0.5 * nodes_std + 0.5
    weights = 0.5 * weights_std
    
    # We need two sets of perturbations: positive (right) and negative (left)
    delta_right = points / (1 - points)      # Maps (0, 1) -> (0, inf)
    delta_left = -delta_right

    # Combine deltas and corresponding weights for easier processing
    # Shape of all_deltas will be (2 * n_points,)
    all_deltas = np.concatenate([delta_left, delta_right])
    
    # The change-of-variable term in the weights needs to be calculated
    # for both left and right sides.
    # Note: abs(-weights / (1-points)**2) is the same as weights / (1-points)**2
    # So we can just concatenate the weights.
    base_weights = weights / (1 - points)**2
    all_base_weights = np.concatenate([base_weights, base_weights]) # Shape (2 * n_points,)

    # --- Step 1: Calculate all ratios ---
    # This involves creating a large batch of points to evaluate f efficiently.
    total_evals = sample_len * dim * (2 * n_points)
    base_X = np.repeat(X, dim * (2 * n_points), axis=0)

    perturbation_matrix = np.zeros_like(base_X)
    delta_values = np.tile(np.tile(all_deltas, dim), sample_len)
    row_indices = np.arange(total_evals)
    col_indices = np.tile(np.repeat(np.arange(dim), (2 * n_points)), sample_len)
    perturbation_matrix[row_indices, col_indices] = delta_values
    
    eval_batch = base_X + perturbation_matrix
    f_values_perturbed_flat = f(eval_batch)

    # Reshape to (sample_len, dim, 2 * n_points) for easy processing
    f_values_perturbed = f_values_perturbed_flat.reshape(sample_len, dim, 2 * n_points)
    
    # Calculate the finite difference ratios
    y_perturbed = f_values_perturbed - f_value_old[:, np.newaxis, np.newaxis]
    # Use np.divide to handle potential division by zero safely
    ratios = np.divide(y_perturbed, all_deltas, out=np.zeros_like(y_perturbed), where=all_deltas != 0)
    
    # --- Step 2: Construct the matrix C by summing ratios with their weights ---
    
    # First, calculate the full weights, including the PDF part
    perturbed_coords = eval_batch[row_indices, col_indices].reshape(sample_len, dim, 2 * n_points)
    pdf_weights = stats.norm.pdf(perturbed_coords)
    
    all_weights = all_base_weights[np.newaxis, np.newaxis, :] * pdf_weights
    
    # Normalize weights for each sample and each dimension so they sum to 1
    # This completes the expectation (integral approximation)
    weight_sum = np.sum(all_weights, axis=2, keepdims=True)
    all_weights = np.divide(all_weights, weight_sum, out=np.zeros_like(all_weights), where=weight_sum!=0)
    
    # -- Calculate matrix elements --
    
    # For off-diagonal C_ij = E[ratio_i] * E[ratio_j]
    # We first need E[ratio_k] for all dimensions k
    mean_ratios = np.sum(ratios * all_weights, axis=2) # Shape: (sample_len, dim)
    
    # Use broadcasting to compute the outer product for each sample
    # C[s, i, j] = mean_ratios[s, i] * mean_ratios[s, j]
    C_off_diagonal = mean_ratios[:, :, np.newaxis] * mean_ratios[:, np.newaxis, :]
    
    # For diagonal C_ii = E[(ratio_i)^2]
    # We calculate the weighted sum of the squared ratios
    mean_squared_ratios = np.sum((ratios**2) * all_weights, axis=2) # Shape: (sample_len, dim)

    # Create the final matrix C. Start with the off-diagonal terms.
    C = C_off_diagonal
    
    # Now, fill in the diagonal with the correct values.
    # We can use a trick with einsum or a simple loop. A loop is very clear.
    for i in range(dim):
        C[:, i, i] = mean_squared_ratios[:, i]
        
    return C.mean(axis=0)

def calculate_gamma(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    U: np.ndarray,
    n_points: int = 8  # Higher n_points for better accuracy
) -> np.ndarray:
    """
    Calculates the Gamma_i values for i=1,...,d using a combination of
    Monte Carlo (for the outer expectation) and Gaussian-Legendre quadrature
    (for the inner integral).
    """
    sample_len, dim = X.shape
    f_value_old = f(X)

    # --- Quadrature Setup ---
    nodes_std, weights_std = leggauss(n_points)
    points = 0.5 * nodes_std + 0.5
    weights = 0.5 * weights_std
    delta_right = points / (1 - points)
    delta_left = -delta_right
    all_deltas = np.concatenate([delta_left, delta_right])
    base_weights = weights / (1 - points)**2
    all_base_weights = np.concatenate([base_weights, base_weights])

    # --- Step 1: Batched Evaluation ---
    X_expanded = X[:, np.newaxis, np.newaxis, :]
    U_T_expanded = U.T[np.newaxis, :, np.newaxis, :]
    deltas_expanded = all_deltas[np.newaxis, np.newaxis, :, np.newaxis]

    perturbations = deltas_expanded * U_T_expanded
    eval_batch_4d = X_expanded + perturbations
    eval_batch_flat = eval_batch_4d.reshape(-1, dim)
    
    f_values_perturbed_flat = f(eval_batch_flat)
    f_values_perturbed = f_values_perturbed_flat.reshape(sample_len, dim, 2 * n_points)

    # --- Step 2: Calculate Squared Ratios (FIXED LINE) ---
    y_perturbed = f_values_perturbed - f_value_old[:, np.newaxis, np.newaxis]
    ratios_sq = np.divide(y_perturbed, all_deltas, out=np.zeros_like(y_perturbed), where=all_deltas != 0)**2

    # --- Step 3: Calculate Quadrature Weights ---
    w = X @ U
    v_w_values = w[:, :, np.newaxis] + all_deltas[np.newaxis, np.newaxis, :]
    
    pdf_weights = stats.norm.pdf(v_w_values)
    all_weights = all_base_weights[np.newaxis, np.newaxis, :] * pdf_weights
    
    weight_sum = np.sum(all_weights, axis=2, keepdims=True)
    all_weights = np.divide(all_weights, weight_sum, out=np.zeros_like(all_weights), where=weight_sum != 0)

    # --- Step 4: Compute Final Result ---
    inner_integrals = np.sum(ratios_sq * all_weights, axis=2)
    gamma = np.mean(inner_integrals, axis=0)

    return gamma

if __name__ == '__main__':
    def f_test(x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        return x1 + 3 * x2

    X_test = np.array([
        [0.1, 0.2],
        [0.5, 0.3],
        [0.8, 0.4]
    ])
    
    # --- Run the numerical calculation with different n_points ---
    for points in [1,4,10]:
        print(f"--- Calculating with n_points = {points} ---")
        fd_result = calculate_sensitivity_matrix(
            f=f_test,
            X=X_test,
            n_points=points
        )
        print(f"Resulting finite difference matrix:\n{fd_result}\n")

    print("Note:")
    print("The 2-point rule is exact for the second dimension because f is linear in x2.")
    print("The 3-point rule (Milne's) is exact for polynomials of degree 3 or less, so it is exact for x1^2.")



