"""Random Search Optimizers
==========================

Simple random and grid search for baseline comparison.

Classes:
    RandomSearchOptimizer: Sample random points within bounds
    GridSearchOptimizer: Exhaustive grid search

Useful for:
    - Baseline comparison (CMA-ES should beat random)
    - Quick sanity checks
    - Understanding loss landscape

Example:
    from hh_envelope.learning import RandomSearchOptimizer

    optimizer = RandomSearchOptimizer(bounds, num_samples=200, seed=42)
    result = optimizer.optimize(objective_fn)
"""

from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np

from hh_envelope.core import HEUThetaBounds

from .optim_cmaes import THETA_NAMES, OptimizeResult


class RandomSearchOptimizer:
    """Random search optimizer for baseline comparison.

    Samples random points uniformly within the bounds and
    returns the best one found. Useful for:
    - Baseline comparison (other methods should beat random)
    - Quick sanity checks
    - Exploring the loss landscape
    """

    def __init__(
        self,
        bounds: HEUThetaBounds | None = None,
        num_samples: int = 100,
        seed: int | None = None,
    ):
        """Initialize the random search optimizer.

        Args:
            bounds: Bounds for theta parameters
            num_samples: Number of random samples to evaluate
            seed: Random seed for reproducibility
        """
        self._bounds = bounds or HEUThetaBounds()
        self._num_samples = num_samples
        self._rng = np.random.default_rng(seed)

        # Compute bounds arrays
        self._lower_bounds = np.array(
            [
                self._bounds.theta_total[0],
                self._bounds.theta_balance[0],
                self._bounds.theta_attack_tau[0],
                self._bounds.theta_recovery_tau[0],
            ]
        )
        self._upper_bounds = np.array(
            [
                self._bounds.theta_total[1],
                self._bounds.theta_balance[1],
                self._bounds.theta_attack_tau[1],
                self._bounds.theta_recovery_tau[1],
            ]
        )

    def optimize(
        self,
        objective_fn: Callable[[dict[str, float]], float],
        verbose: bool = False,
    ) -> OptimizeResult:
        """Run random search optimization.

        Args:
            objective_fn: Function mapping theta dict to loss value
            verbose: Print progress

        Returns:
            OptimizeResult with best parameters found
        """
        best_loss = float("inf")
        best_theta_arr = None
        loss_history = []

        for i in range(self._num_samples):
            # Sample uniformly within bounds
            theta_arr = self._rng.uniform(self._lower_bounds, self._upper_bounds)
            theta_dict = self._theta_array_to_dict(theta_arr)

            # Evaluate
            loss = objective_fn(theta_dict)

            if loss < best_loss:
                best_loss = loss
                best_theta_arr = theta_arr.copy()

            loss_history.append(best_loss)

            if verbose and (i + 1) % 10 == 0:
                print(f"Sample {i + 1}/{self._num_samples}: best_loss = {best_loss:.6f}")

        if best_theta_arr is None:
            best_theta_arr = self._rng.uniform(self._lower_bounds, self._upper_bounds)

        return OptimizeResult(
            best_theta=self._theta_array_to_dict(best_theta_arr),
            best_loss=best_loss,
            iterations=self._num_samples,
            loss_history=loss_history,
            converged=False,  # Random search doesn't "converge"
            evaluations=self._num_samples,
        )

    def _theta_array_to_dict(self, arr: np.ndarray) -> dict[str, float]:
        """Convert optimizer array to theta dictionary."""
        return {name: float(arr[i]) for i, name in enumerate(THETA_NAMES)}


class GridSearchOptimizer:
    """Grid search optimizer for systematic exploration.

    Evaluates a regular grid of points within the bounds.
    Exponentially expensive in dimension, but gives complete
    coverage of the search space.
    """

    def __init__(
        self,
        bounds: HEUThetaBounds | None = None,
        resolution: int = 5,
    ):
        """Initialize the grid search optimizer.

        Args:
            bounds: Bounds for theta parameters
            resolution: Number of points per dimension
        """
        self._bounds = bounds or HEUThetaBounds()
        self._resolution = resolution

        # Compute bounds arrays
        self._lower_bounds = np.array(
            [
                self._bounds.theta_total[0],
                self._bounds.theta_balance[0],
                self._bounds.theta_attack_tau[0],
                self._bounds.theta_recovery_tau[0],
            ]
        )
        self._upper_bounds = np.array(
            [
                self._bounds.theta_total[1],
                self._bounds.theta_balance[1],
                self._bounds.theta_attack_tau[1],
                self._bounds.theta_recovery_tau[1],
            ]
        )

        # Pre-compute grid
        self._grid_points = self._compute_grid()

    def _compute_grid(self) -> list[np.ndarray]:
        """Compute all grid points."""
        dim = len(self._lower_bounds)
        ranges = [
            np.linspace(self._lower_bounds[i], self._upper_bounds[i], self._resolution)
            for i in range(dim)
        ]
        return [np.array(point) for point in product(*ranges)]

    @property
    def num_points(self) -> int:
        """Total number of grid points."""
        return len(self._grid_points)

    def optimize(
        self,
        objective_fn: Callable[[dict[str, float]], float],
        verbose: bool = False,
    ) -> OptimizeResult:
        """Run grid search optimization.

        Args:
            objective_fn: Function mapping theta dict to loss value
            verbose: Print progress

        Returns:
            OptimizeResult with best parameters found
        """
        best_loss = float("inf")
        best_theta_arr = None
        loss_history = []
        all_results = []

        for i, theta_arr in enumerate(self._grid_points):
            theta_dict = self._theta_array_to_dict(theta_arr)

            # Evaluate
            loss = objective_fn(theta_dict)
            all_results.append((theta_arr.copy(), loss))

            if loss < best_loss:
                best_loss = loss
                best_theta_arr = theta_arr.copy()

            loss_history.append(best_loss)

            if verbose and (i + 1) % 50 == 0:
                print(f"Point {i + 1}/{len(self._grid_points)}: best_loss = {best_loss:.6f}")

        if best_theta_arr is None:
            best_theta_arr = (self._lower_bounds + self._upper_bounds) / 2

        return OptimizeResult(
            best_theta=self._theta_array_to_dict(best_theta_arr),
            best_loss=best_loss,
            iterations=len(self._grid_points),
            loss_history=loss_history,
            converged=False,  # Grid search doesn't "converge"
            evaluations=len(self._grid_points),
        )

    def _theta_array_to_dict(self, arr: np.ndarray) -> dict[str, float]:
        """Convert optimizer array to theta dictionary."""
        return {name: float(arr[i]) for i, name in enumerate(THETA_NAMES)}

    def get_loss_landscape(
        self,
        objective_fn: Callable[[dict[str, float]], float],
        dimensions: tuple[int, int] = (0, 1),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get 2D slice of loss landscape for visualization.

        Evaluates grid on two selected dimensions while holding
        others at center of bounds.

        Args:
            objective_fn: Function mapping theta dict to loss value
            dimensions: Indices of two dimensions to vary

        Returns:
            Tuple of (X grid, Y grid, loss values)
        """
        d1, d2 = dimensions

        # Create 2D grid
        x_range = np.linspace(self._lower_bounds[d1], self._upper_bounds[d1], self._resolution)
        y_range = np.linspace(self._lower_bounds[d2], self._upper_bounds[d2], self._resolution)

        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)

        # Center point for other dimensions
        center = (self._lower_bounds + self._upper_bounds) / 2

        for i in range(self._resolution):
            for j in range(self._resolution):
                theta_arr = center.copy()
                theta_arr[d1] = X[i, j]
                theta_arr[d2] = Y[i, j]

                theta_dict = self._theta_array_to_dict(theta_arr)
                Z[i, j] = objective_fn(theta_dict)

        return X, Y, Z
