"""CMA-ES Optimizer
==================

Wrapper around the `cma` library for envelope parameter optimization.

The CMAESOptimizer provides automatic bounds handling, configurable
population size, and conversion between theta arrays and dictionaries.

Example:
    from hh_envelope.learning import CMAESOptimizer
    from hh_envelope.core import HEUThetaBounds

    bounds = HEUThetaBounds()
    optimizer = CMAESOptimizer(bounds, sigma0=0.5)

    result = optimizer.optimize(objective_fn, max_iterations=100)
    print(f"Best loss: {result.best_loss}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from hh_envelope.core import HEUThetaBounds


@dataclass
class OptimizeResult:
    """Result of an optimization run."""

    best_theta: dict[str, float]
    """Best theta parameters found."""

    best_loss: float
    """Loss value at best parameters."""

    iterations: int
    """Number of iterations completed."""

    loss_history: list[float] = field(default_factory=list)
    """History of best loss at each iteration."""

    converged: bool = False
    """Whether optimization converged before max iterations."""

    evaluations: int = 0
    """Total number of function evaluations."""


# Parameter order for array conversion
THETA_NAMES = ["theta_total", "theta_balance", "theta_attack_tau", "theta_recovery_tau"]


class CMAESOptimizer:
    """CMA-ES optimizer for HEU parameter fitting.

    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a
    derivative-free optimization algorithm well-suited for:
    - Low-dimensional problems (< 10 params)
    - Non-convex objectives
    - Noisy objective functions
    """

    def __init__(
        self,
        bounds: HEUThetaBounds | None = None,
        sigma0: float = 0.5,
        popsize: int | None = None,
        seed: int | None = None,
    ):
        """Initialize the CMA-ES optimizer.

        Args:
            bounds: Bounds for theta parameters
            sigma0: Initial step size (standard deviation)
            popsize: Population size (None = default based on dimension)
            seed: Random seed for reproducibility
        """
        self._bounds = bounds or HEUThetaBounds()
        self._sigma0 = sigma0
        self._popsize = popsize
        self._seed = seed

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

        # Initial point: center of bounds
        self._x0 = (self._lower_bounds + self._upper_bounds) / 2

    def optimize(
        self,
        objective_fn: Callable[[dict[str, float]], float],
        max_iterations: int = 100,
        target_loss: float | None = None,
        verbose: bool = False,
    ) -> OptimizeResult:
        """Run CMA-ES optimization.

        Args:
            objective_fn: Function mapping theta dict to loss value
            max_iterations: Maximum number of generations
            target_loss: Stop early if loss drops below this
            verbose: Print progress

        Returns:
            OptimizeResult with best parameters and history
        """
        try:
            import cma  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("CMA-ES requires the 'cma' package. Install with: pip install cma")

        # Build CMA-ES options
        opts = {
            "bounds": [self._lower_bounds.tolist(), self._upper_bounds.tolist()],
            "seed": self._seed,
            "verbose": -9 if not verbose else 1,  # Suppress output unless verbose
        }
        if self._popsize is not None:
            opts["popsize"] = self._popsize

        # Create optimizer
        es = cma.CMAEvolutionStrategy(self._x0.tolist(), self._sigma0, opts)

        # Tracking
        loss_history = []
        best_loss = float("inf")
        best_theta_arr = self._x0.copy()
        evaluations = 0
        converged = False

        # Optimization loop
        iteration = 0
        while not es.stop() and iteration < max_iterations:
            # Get candidate solutions
            candidates = es.ask()

            # Evaluate each candidate
            losses = []
            for candidate in candidates:
                theta_dict = self.theta_array_to_dict(np.array(candidate))
                loss = objective_fn(theta_dict)
                losses.append(loss)
                evaluations += 1

                if loss < best_loss:
                    best_loss = loss
                    best_theta_arr = np.array(candidate)

            # Update CMA-ES
            es.tell(candidates, losses)

            # Track progress
            loss_history.append(best_loss)

            if verbose:
                print(f"Iteration {iteration}: best_loss = {best_loss:.6f}")

            # Check for early stopping
            if target_loss is not None and best_loss <= target_loss:
                converged = True
                break

            iteration += 1

        return OptimizeResult(
            best_theta=self.theta_array_to_dict(best_theta_arr),
            best_loss=best_loss,
            iterations=iteration,
            loss_history=loss_history,
            converged=converged,
            evaluations=evaluations,
        )

    def theta_array_to_dict(self, arr: np.ndarray) -> dict[str, float]:
        """Convert optimizer array to theta dictionary.

        Args:
            arr: Array of theta values in standard order

        Returns:
            Dictionary with named theta parameters
        """
        return {name: float(arr[i]) for i, name in enumerate(THETA_NAMES)}

    def theta_dict_to_array(self, theta_dict: dict[str, float]) -> np.ndarray:
        """Convert theta dictionary to optimizer array.

        Args:
            theta_dict: Dictionary with named theta parameters

        Returns:
            Array of theta values in standard order
        """
        return np.array([theta_dict.get(name, 0.0) for name in THETA_NAMES])
