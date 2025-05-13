#!/usr/bin/env python
"""Quick script to verify Anderson Acceleration test convergence rate"""

import platform
import numpy as np
from numpy.linalg import norm
import sys
from skglm.utils.anderson import AndersonAcceleration
import numba
import sklearn


def check_convergence():
    # Same setup as in test_anderson_acceleration
    max_iter, tol = 1000, 1e-9
    n_features = 2
    rho = np.array([0.5, 0.8])
    w_star = 1 / (1 - rho)
    X = np.diag([2, 5])

    # With acceleration
    acc = AndersonAcceleration(K=5)
    n_iter_acc = 0
    w = np.ones(n_features)
    Xw = X @ w
    errors_acc = []

    for i in range(max_iter):
        w, Xw, _ = acc.extrapolate(w, Xw)
        w = rho * w + 1
        Xw = X @ w
        errors_acc.append(norm(w - w_star, ord=np.inf))

        if norm(w - w_star, ord=np.inf) < tol:
            n_iter_acc = i
            break

    # Without acceleration
    n_iter = 0
    w = np.ones(n_features)
    errors = []
    for i in range(max_iter):
        w = rho * w + 1
        errors.append(norm(w - w_star, ord=np.inf))
        if norm(w - w_star, ord=np.inf) < tol:
            n_iter = i
            break

    return n_iter_acc, n_iter, errors_acc, errors


if __name__ == "__main__":
    n_iter_acc, n_iter, errors_acc, errors = check_convergence()

    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"Numba: {numba.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"Float precision: {np.finfo(np.float64)}")

    # Get BLAS info safely
    try:
        blas_info = np.show_config()
        print("\nBLAS/LAPACK Configuration:")
        print(blas_info)
    except Exception:
        print("\nCould not retrieve BLAS/LAPACK info")

    print(f"\nConvergence Results:")
    print(f"Iterations with acceleration: {n_iter_acc} (expected: 13)")
    print(f"Iterations without acceleration: {n_iter}")
    print(f"Speedup factor: {n_iter/n_iter_acc:.2f}x")

    print("\nConvergence Rate Analysis:")
    print("With acceleration:")
    print(f"Error at iteration {n_iter_acc-1}: {errors_acc[n_iter_acc-1]:.2e}")
    print(f"Error at iteration {n_iter_acc}: {errors_acc[n_iter_acc]:.2e}")

    if n_iter_acc != 13:
        print(f"\nTest fails: converges in {n_iter_acc} iterations instead of 13")
        print("Potential reasons:")
        print("1. Different BLAS/LAPACK implementation affecting matrix operations")
        print("2. Different floating-point precision behavior")
        print("3. Different convergence path due to numerical differences")
