import numpy as np
from typing import Callable
from scipy.integrate import newton_cotes
from scipy.special import roots_legendre

NEWTON_COTES_POINTS = {
    1: np.array([1/2]),
    2: np.array([1/3, 2/3]),
    3: np.array([1/4, 2/4, 3/4]),
    4: np.array([1/5, 2/5, 3/5, 4/5]),
}

NEWTON_COTES_WEIGHTS = {
    1: np.array([1.0]),
    2: np.array([1/2, 1/2]),
    3: np.array([1/6, 4/6, 1/6]),
    4: np.array([1/8, 3/8, 3/8, 1/8]),
}

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

    #weights = NEWTON_COTES_WEIGHTS[n_points]
    #points = NEWTON_COTES_POINTS[n_points]
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

if __name__ == '__main__':
    def f_test(x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        return x1 + 2 * x2

    X_test = np.array([
        [0.1, 0.2],
        [0.5, 0.3],
        [0.8, 0.4]
    ])
    
    # --- Run the numerical calculation with different n_points ---
    for points in [4,10]:
        print(f"--- Calculating with n_points = {points} ---")
        fd_result = calculate_finite_difference_matrix(
            f=f_test,
            X=X_test,
            n_points=points
        )
        print(f"Resulting finite difference matrix:\n{fd_result}\n")

    print("Note:")
    print("The 2-point rule is exact for the second dimension because f is linear in x2.")
    print("The 3-point rule (Milne's) is exact for polynomials of degree 3 or less, so it is exact for x1^2.")