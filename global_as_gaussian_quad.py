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
    delta_right = points      # for uniform, (0, 1) is enough
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


import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Callable
def calculate_sensitivity_matrix_uniform(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_points: int = 4,
    min_delta: float = 1e-3  # Threshold: specific steps smaller than this are masked
) -> np.ndarray:
    """
    Calculates sensitivity matrix C using Gauss-Legendre Split method.
    
    Refinement: Instead of discarding entire intervals, it masks individual 
    quadrature nodes where the perturbation (delta) is too small.
    This prevents the noise term (1/delta^2) from exploding while preserving
    integral contributions from safe nodes.
    """
    sample_len, dim = X.shape
    f_value_old = f(X)

    # --- 1. Quadrature Setup ---
    nodes_std, weights_std = leggauss(n_points)
    base_weights = weights_std / 2.0 

    # --- 2. Calculate Mappings (Standard Split) ---
    X_expanded = X[:, :, np.newaxis] 
    
    # -- Left Interval [0, X] --
    points_left = (X_expanded / 2.0) * (nodes_std + 1)
    deltas_left = points_left - X_expanded
    weights_left_scaled = base_weights * X_expanded 

    # -- Right Interval [X, 1] --
    points_right = ((1 - X_expanded) / 2.0) * nodes_std + ((1 + X_expanded) / 2.0)
    deltas_right = points_right - X_expanded
    weights_right_scaled = base_weights * (1 - X_expanded)

    # Combine
    all_deltas = np.concatenate([deltas_left, deltas_right], axis=2)
    all_weights = np.concatenate([weights_left_scaled, weights_right_scaled], axis=2)

    # --- 3. Apply Fine-Grained Masking ---
    # This is the key change. We check the absolute size of every single delta.
    # If delta is too small, the ratio noise will be huge. We mask THIS weight to 0.
    
    mask = np.abs(all_deltas) > min_delta
    
    # Apply mask to weights. 
    # Invalid nodes now have weight 0.0, so they contribute nothing to the sum.
    all_weights = all_weights * mask

    # (Optional) Re-normalization could be done here if you wanted to maintain 
    # exact probability mass, but for noise suppression, simply dropping the 
    # exploding term is usually safer and sufficient.

    # --- 4. Batched Evaluation ---
    total_perturbations = sample_len * dim * (2 * n_points)
    base_X = np.repeat(X, dim * (2 * n_points), axis=0)
    perturbation_matrix = np.zeros_like(base_X)
    
    delta_values_flat = all_deltas.reshape(sample_len, -1).flatten()
    row_indices = np.arange(total_perturbations)
    col_indices = np.tile(np.repeat(np.arange(dim), 2 * n_points), sample_len)
    
    perturbation_matrix[row_indices, col_indices] = delta_values_flat
    
    eval_batch = base_X + perturbation_matrix
    
    f_values_perturbed = f(eval_batch).reshape(sample_len, dim, 2 * n_points)
    
    # --- 5. Slopes and Integration ---
    y_diff = f_values_perturbed - f_value_old[:, np.newaxis, np.newaxis]
    
    # Safe Division:
    # We only care about division where mask is True.
    # However, to avoid RuntimeWarnings, we use the 'where' argument.
    # We reuse 'mask' which tells us exactly where delta is safe.
    ratios = np.zeros_like(y_diff)
    np.divide(y_diff, all_deltas, out=ratios, where=mask)
    
    # Gradient Estimate E[g]
    # Because all_weights is 0 where mask is False, the "bad" ratios are ignored.
    gradient_estimate = np.sum(ratios * all_weights, axis=2)
    
    # Second Moment Estimate E[g^2]
    gradient_squared_estimate = np.sum((ratios**2) * all_weights, axis=2)
    
    # --- 6. Construct Matrix C ---
    C = gradient_estimate[:, :, np.newaxis] * gradient_estimate[:, np.newaxis, :]
    
    for i in range(dim):
        C[:, i, i] = gradient_squared_estimate[:, i]
        
    return C.mean(axis=0)

import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Callable

def calculate_sensitivity_matrix_renormalized(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_points: int = 5,
    min_delta: float = 0.05
) -> np.ndarray:
    """
    Calculates sensitivity matrix C using Gauss-Legendre Split method.
    
    Refinement: Point-wise Masking with Weight Re-normalization.
    1. Masks specific nodes that are too close to X.
    2. Scales up the weights of the REMAINING nodes in that interval so that
       the sum of weights still equals the interval length.
    """
    sample_len, dim = X.shape
    f_value_old = f(X)

    # --- 1. Quadrature Setup ---
    nodes_std, weights_std = leggauss(n_points)
    base_weights = weights_std / 2.0 
    X_expanded = X[:, :, np.newaxis] 

    # --- 2. Calculate Mappings (Left and Right Separately) ---
    # We keep them separate initially to normalize them independently.

    # -- Left Interval [0, X] --
    points_left = (X_expanded / 2.0) * (nodes_std + 1)
    deltas_left = points_left - X_expanded
    weights_left = base_weights * X_expanded 
    # Target sum for Left weights is exactly the length: X
    target_sum_left = X_expanded

    # -- Right Interval [X, 1] --
    points_right = ((1 - X_expanded) / 2.0) * nodes_std + ((1 + X_expanded) / 2.0)
    deltas_right = points_right - X_expanded
    weights_right = base_weights * (1 - X_expanded)
    # Target sum for Right weights is exactly the length: 1 - X
    target_sum_right = 1.0 - X_expanded

    # --- 3. Masking & Re-normalization ---

    def renormalize_weights(deltas, weights, target_sum):
        # 1. Identify safe nodes
        mask = np.abs(deltas) > min_delta
        
        # 2. Apply mask (unsafe nodes become 0.0)
        safe_weights = weights * mask
        
        # 3. Calculate the actual sum of weights we have left
        # Shape: (sample_len, dim, 1)
        actual_sum = np.sum(safe_weights, axis=2, keepdims=True)
        
        # 4. Calculate Scaling Factor
        # If actual_sum is 0 (all nodes masked), we can't scale. set factor to 0.
        factor = np.zeros_like(actual_sum)
        
        # Safe division: target / actual
        np.divide(target_sum, actual_sum, out=factor, where=actual_sum > 0)
        
        # 5. Scale the weights
        # The remaining nodes now carry the burden of the masked nodes
        return safe_weights * factor

    # Apply to Left
    weights_left_final = renormalize_weights(deltas_left, weights_left, target_sum_left)
    
    # Apply to Right
    weights_right_final = renormalize_weights(deltas_right, weights_right, target_sum_right)

    # --- 4. Combine & Evaluate ---
    all_deltas = np.concatenate([deltas_left, deltas_right], axis=2)
    all_weights = np.concatenate([weights_left_final, weights_right_final], axis=2)

    total_perturbations = sample_len * dim * (2 * n_points)
    base_X = np.repeat(X, dim * (2 * n_points), axis=0)
    perturbation_matrix = np.zeros_like(base_X)
    
    delta_values_flat = all_deltas.reshape(sample_len, -1).flatten()
    row_indices = np.arange(total_perturbations)
    col_indices = np.tile(np.repeat(np.arange(dim), 2 * n_points), sample_len)
    
    perturbation_matrix[row_indices, col_indices] = delta_values_flat
    
    eval_batch = base_X + perturbation_matrix
    f_values_perturbed = f(eval_batch).reshape(sample_len, dim, 2 * n_points)
    
    # --- 5. Gradient Estimation ---
    y_diff = f_values_perturbed - f_value_old[:, np.newaxis, np.newaxis]
    
    # Safe Division (only where we have non-zero weights)
    # If a weight was masked to 0, we don't care about the ratio.
    valid_mask = all_weights > 0.0
    ratios = np.zeros_like(y_diff)
    np.divide(y_diff, all_deltas, out=ratios, where=valid_mask)
    
    # Gradient Estimate E[g]
    gradient_estimate = np.sum(ratios * all_weights, axis=2)
    
    # Second Moment Estimate E[g^2]
    gradient_squared_estimate = np.sum((ratios**2) * all_weights, axis=2)
    
    # --- 6. Construct Matrix C ---
    C = gradient_estimate[:, :, np.newaxis] * gradient_estimate[:, np.newaxis, :]
    
    for i in range(dim):
        C[:, i, i] = gradient_squared_estimate[:, i]
        
    return C.mean(axis=0)

def calculate_sensitivity_matrix_side_mask(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_points: int = 5,
    min_delta: float = 0.05 
) -> np.ndarray:
    """
    Calculates sensitivity matrix C using Gauss-Legendre Split method.
    
    Refinement: Side-Interval Masking.
    - Checks if the Left Interval [0, X] is large enough. If not, weights_left = 0.
    - Checks if the Right Interval [X, 1] is large enough. If not, weights_right = 0.
    - Does NOT drop the sample. It just ignores the contribution from the unstable side.
    """
    sample_len, dim = X.shape
    f_value_old = f(X)

    # --- 1. Quadrature Setup & Safety Threshold ---
    nodes_std, weights_std = leggauss(n_points)
    base_weights = weights_std / 2.0 

    # Calculate the minimum interval length required to satisfy min_delta
    # The closest a node gets to the edge in standard [-1, 1] is max(nodes_std)
    # The distance ratio is (1 - max_node) / 2
    max_node = np.max(nodes_std)
    scaling_factor = (1.0 - max_node) / 2.0
    
    # We need: Length * scaling_factor > min_delta
    min_required_len = min_delta / scaling_factor
    
    # --- 2. Calculate Mappings ---
    X_expanded = X[:, :, np.newaxis] 
    
    # -- Left Interval [0, X] --
    points_left = (X_expanded / 2.0) * (nodes_std + 1)
    deltas_left = points_left - X_expanded
    weights_left = base_weights * X_expanded 

    # -- Right Interval [X, 1] --
    points_right = ((1 - X_expanded) / 2.0) * nodes_std + ((1 + X_expanded) / 2.0)
    deltas_right = points_right - X_expanded
    weights_right = base_weights * (1 - X_expanded)

    # --- 3. Apply Side-Interval Masking ---
    
    # Lengths of intervals
    len_left = X                     # shape (sample_len, dim)
    len_right = 1.0 - X              # shape (sample_len, dim)

    # Create Boolean Masks (True if safe, False if unsafe)
    mask_left = len_left > min_required_len
    mask_right = len_right > min_required_len

    # Expand masks to apply to weights: (sample_len, dim, 1) to broadcast over n_points
    weights_left = weights_left * mask_left[:, :, np.newaxis]
    weights_right = weights_right * mask_right[:, :, np.newaxis]

    # Combine
    all_deltas = np.concatenate([deltas_left, deltas_right], axis=2)
    all_weights = np.concatenate([weights_left, weights_right], axis=2)

    # --- 4. Batched Evaluation ---
    total_perturbations = sample_len * dim * (2 * n_points)
    base_X = np.repeat(X, dim * (2 * n_points), axis=0)
    perturbation_matrix = np.zeros_like(base_X)
    
    delta_values_flat = all_deltas.reshape(sample_len, -1).flatten()
    row_indices = np.arange(total_perturbations)
    col_indices = np.tile(np.repeat(np.arange(dim), 2 * n_points), sample_len)
    
    perturbation_matrix[row_indices, col_indices] = delta_values_flat
    
    eval_batch = base_X + perturbation_matrix
    f_values_perturbed = f(eval_batch).reshape(sample_len, dim, 2 * n_points)
    
    # --- 5. Slopes and Integration ---
    y_diff = f_values_perturbed - f_value_old[:, np.newaxis, np.newaxis]
    
    # Safe Division:
    # We only care about division where weight > 0.
    # If weight is 0 (masked), we don't care about the result (it will be summed as 0).
    # But to avoid RuntimeWarning or NaN from dividing by tiny deltas in masked regions:
    valid_division_mask = all_weights > 0.0
    
    ratios = np.zeros_like(y_diff)
    np.divide(y_diff, all_deltas, out=ratios, where=valid_division_mask)
    
    # Gradient Estimate E[g]
    gradient_estimate = np.sum(ratios * all_weights, axis=2)
    
    # Second Moment Estimate E[g^2]
    gradient_squared_estimate = np.sum((ratios**2) * all_weights, axis=2)
    
    # --- 6. Construct Matrix C ---
    C = gradient_estimate[:, :, np.newaxis] * gradient_estimate[:, np.newaxis, :]
    
    for i in range(dim):
        C[:, i, i] = gradient_squared_estimate[:, i]
        
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



