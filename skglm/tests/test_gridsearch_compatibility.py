# import numpy as np
# import pytest
# import warnings
# from sklearn.model_selection import GridSearchCV
# from sklearn.datasets import make_regression, make_classification

# from skglm.estimators import GeneralizedLinearEstimator
# from skglm.datafits import (
#     Quadratic, Logistic, Huber, Poisson, Gamma,
#     QuadraticMultiTask, QuadraticGroup, WeightedQuadratic, QuadraticSVC
# )
# from skglm.penalties import (
#     L1, L1_plus_L2, L2, WeightedGroupL2, L2_1
# )
# from skglm.solvers import AndersonCD, ProxNewton, MultiTaskBCD, GroupBCD


# def test_gridsearch_regression():
#     """Test GridSearchCV with GeneralizedLinearEstimator for regression."""
#     regression_data = make_regression(n_samples=100, n_features=20, random_state=42)
#     X, y = regression_data[:2]  # Only take X and y, ignore coef

#     # Create base estimator
#     base_estimator = GeneralizedLinearEstimator(
#         datafit=Quadratic(),
#         penalty=L1_plus_L2(alpha=1.0, l1_ratio=0.5),
#         solver=AndersonCD()
#     )

#     # Set appropriate ws_strategy for AndersonCD solver
#     if isinstance(base_estimator.solver, AndersonCD):
#         if hasattr(base_estimator.penalty, 'subdiff_distance'):
#             base_estimator.solver.ws_strategy = 'subdiff'
#         else:
#             base_estimator.solver.ws_strategy = 'fixpoint'

#     # Define parameter grid
#     param_grid = {
#         'penalty__alpha': [0.1, 1.0, 10.0],
#         'penalty__l1_ratio': [0.3, 0.5, 0.7],
#         'solver__tol': [1e-3, 1e-4]
#     }

#     # Create GridSearchCV
#     grid_search = GridSearchCV(
#         base_estimator,
#         param_grid,
#         cv=3,
#         scoring='r2',
#         n_jobs=-1
#     )

#     # Fit GridSearchCV
#     grid_search.fit(X, y)

#     # Check that the best model has reasonable performance
#     assert grid_search.best_score_ > 0.5


# def test_gridsearch_classification():
#     """Test GridSearchCV with GeneralizedLinearEstimator for classification."""
#     X, y = make_classification(n_samples=100, n_features=20, random_state=42)

#     # Create base estimator
#     base_estimator = GeneralizedLinearEstimator(
#         datafit=Logistic(),
#         penalty=L1_plus_L2(alpha=1.0, l1_ratio=0.5),
#         solver=AndersonCD()
#     )

#     # Set appropriate ws_strategy for AndersonCD solver
#     if isinstance(base_estimator.solver, AndersonCD):
#         if hasattr(base_estimator.penalty, 'subdiff_distance'):
#             base_estimator.solver.ws_strategy = 'subdiff'
#         else:
#             base_estimator.solver.ws_strategy = 'fixpoint'

#     # Define parameter grid
#     param_grid = {
#         'penalty__alpha': [0.1, 1.0, 10.0],
#         'penalty__l1_ratio': [0.3, 0.5, 0.7],
#         'solver__tol': [1e-3, 1e-4]
#     }

#     # Create GridSearchCV
#     grid_search = GridSearchCV(
#         base_estimator,
#         param_grid,
#         cv=3,
#         scoring='accuracy',
#         n_jobs=-1
#     )

#     # Fit GridSearchCV
#     grid_search.fit(X, y)

#     # Check that the best model has reasonable performance
#     assert grid_search.best_score_ > 0.5


# @pytest.mark.parametrize('datafit,penalty,solver', [
#     # Regression datafits
#     pytest.param(Quadratic(), L1(alpha=1.0), AndersonCD(),
#                  marks=pytest.mark.regression),
#     pytest.param(Quadratic(), L1_plus_L2(alpha=1.0, l1_ratio=0.5),
#                  AndersonCD(), marks=pytest.mark.regression),
#     pytest.param(Quadratic(), L2(alpha=1.0), AndersonCD(),
#                  marks=pytest.mark.regression),
#     pytest.param(Huber(delta=1.35), L1(alpha=1.0),
#                  AndersonCD(), marks=pytest.mark.regression),
#     pytest.param(Poisson(), L1(alpha=1.0), ProxNewton(), marks=pytest.mark.regression),
#     pytest.param(Gamma(), L1(alpha=1.0), ProxNewton(), marks=pytest.mark.regression),
#     pytest.param(WeightedQuadratic(), L1(
#         alpha=1.0), AndersonCD(), marks=pytest.mark.regression),

#     # Classification datafits
#     pytest.param(Logistic(), L1(alpha=1.0), AndersonCD(),
#                  marks=pytest.mark.classification),
#     pytest.param(Logistic(), L1_plus_L2(alpha=1.0, l1_ratio=0.5),
#                  AndersonCD(), marks=pytest.mark.classification),
#     pytest.param(Logistic(), L2(alpha=1.0), AndersonCD(),
#                  marks=pytest.mark.classification),

#     # Multi-task datafits
#     pytest.param(QuadraticMultiTask(), L2_1(alpha=1.0),
#                  MultiTaskBCD(), marks=pytest.mark.multitask),

#     # Group datafits
#     pytest.param(
#         QuadraticGroup(
#             np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
#                      6, 7, 7, 8, 8, 9, 9], dtype=np.int32),
#             np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
#                      13, 14, 15, 16, 17, 18, 19], dtype=np.int32)
#         ),
#         WeightedGroupL2(
#             alpha=1.0,
#             weights=np.array([1.0] * 10, dtype=np.float64),
#             grp_ptr=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=np.int32),
#             grp_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#                                  11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.int32)
#         ),
#         GroupBCD(),
#         marks=pytest.mark.group
#     ),
# ])
# def test_gridsearch_all_combinations(datafit, penalty, solver):
#     """Test GridSearchCV with different combinations of datafits and penalties."""
#     # Generate appropriate data based on datafit type
#     if isinstance(datafit, QuadraticMultiTask):
#         regression_data = make_regression(n_samples=100, n_features=20,
#                                           n_targets=2, random_state=42)
#         X, y = regression_data[:2]  # Only take X and y, ignore coef
#         y = y.reshape(-1, 2)  # Reshape y to match expected dimensions
#     elif isinstance(datafit, (Logistic, QuadraticSVC)):
#         X, y = make_classification(n_samples=100, n_features=20, random_state=42)
#     elif isinstance(datafit, QuadraticGroup):
#         # For group datafit, ensure data matches group structure
#         n_samples = 100
#         n_features = 20
#         n_groups = 10
#         X = np.random.randn(n_samples, n_features)
#         # Generate coefficients that respect group structure
#         coef = np.zeros(n_features)
#         for g in range(n_groups):
#             coef[2*g:2*g+2] = np.random.randn(2)  # Same coefficient for each group
#         y = X @ coef + np.random.randn(n_samples) * 0.1

#         # Initialize the datafit with the correct group structure
#         grp_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#                                11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.int32)
#         grp_ptr = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=np.int32)
#         datafit = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
#         datafit.initialize(X, y)  # Initialize to compute lipschitz constants
#     else:
#         # For single-task regression, we only need X and y
#         regression_data = make_regression(n_samples=100, n_features=20, random_state=42)
#         X, y = regression_data[:2]  # Only take X and y, ignore coef

#         if isinstance(datafit, (Poisson, Gamma)):
#             # fixed, moderate coefficients for strong signal
#             coef = np.ones(X.shape[1]) * 0.2
#             lin_pred = X @ coef
#             lam = np.exp(lin_pred)
#             lam = np.clip(lam, 1, 5)  # keep lambda in a narrow, reasonable range
#             if isinstance(datafit, Poisson):
#                 y = np.random.poisson(lam)
#                 y = np.maximum(y, 1)  # ensure strictly positive if needed
#             else:  # Gamma
#                 y = lam  # Gamma regression expects positive continuous targets
#         else:
#             y = y.reshape(-1)  # Ensure y is 1D for other single-task problems

#     # Create base estimator
#     base_estimator = GeneralizedLinearEstimator(
#         datafit=datafit,
#         penalty=penalty,
#         solver=solver
#     )

#     # Set appropriate ws_strategy for AndersonCD solver
#     if isinstance(solver, AndersonCD):
#         if hasattr(penalty, 'subdiff_distance'):
#             solver.ws_strategy = 'subdiff'
#         else:
#             solver.ws_strategy = 'fixpoint'

#     # Define parameter grid based on penalty type
#     if isinstance(penalty, WeightedGroupL2):
#         param_grid = {
#             'penalty__alpha': [0.01, 0.1, 1.0],  # Smaller alpha values for group lasso
#             'solver__tol': [1e-3, 1e-4]
#         }
#     else:
#         param_grid = {
#             'penalty__alpha': [0.1, 1.0, 10.0],
#             'penalty__l1_ratio': [0.3, 0.5, 0.7],
#             'solver__tol': [1e-3, 1e-4]
#         }

#     # Choose appropriate scoring metric based on datafit type
#     if isinstance(datafit, Poisson):
#         scoring = 'neg_mean_poisson_deviance'
#         # For deviance metrics, lower (more negative) is better
#         def score_expectation(s): return s < 0
#     elif isinstance(datafit, Gamma):
#         scoring = 'neg_mean_gamma_deviance'
#         def score_expectation(s): return s < 0
#     elif isinstance(datafit, (Logistic, QuadraticSVC)):
#         scoring = 'accuracy'
#         def score_expectation(s): return s > 0.5
#     else:
#         # Default for regression tasks
#         scoring = 'r2'
#         def score_expectation(s): return s > 0.5

#     # Create GridSearchCV with appropriate scoring
#     grid_search = GridSearchCV(
#         base_estimator,
#         param_grid,
#         cv=3,
#         scoring=scoring,
#         n_jobs=-1
#     )

#     fit_params = {}
#     if isinstance(datafit, WeightedQuadratic):
#         fit_params['sample_weight'] = np.ones(X.shape[0])

#     # Fit and validate with warning capture
#     with warnings.catch_warnings(record=True):
#         warnings.simplefilter("always")
#         grid_search.fit(X, y, **fit_params)

#     # Final assertion
#     assert score_expectation(grid_search.best_score_)
