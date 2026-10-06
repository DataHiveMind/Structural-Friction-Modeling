"""
src/simulation/model.py

Mathematical engines for structural market frictions:
1. Merton Jump-Diffusion simulator for latent unobserved price P*_t during trading halts.
2. Liquidity-Adjusted Value-at-Risk (L-VaR) and Expected Shortfall (L-CVaR) modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import scipy.stats as stats


@dataclass(frozen=True)
class JumpDiffusionParams:
    """Parameters governing the jump-diffusion latent price dynamics."""
    drift: float                    # mu(X_t): exogenous macro-conditioned drift
    diffusion_vol: float            # sigma(X_t): continuous Brownian motion volatility
    jump_intensity: float           # lambda: Poisson arrival rate of discrete shocks
    jump_mean: float                # mu_J: expected log-jump size
    jump_std: float                 # sigma_J: standard deviation of log-jump size


class LatentPriceSimulator:
    """
    Simulates the unobserved continuous-time efficient price P*_t during an exchange halt:
        dP*_t = mu(X_t) dt + sigma(X_t) dW_t + J_t dN_t
    """

    def __init__(self, params: JumpDiffusionParams, seed: Optional[int] = None) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)

    def simulate_path(
        self,
        p0: float,
        duration_seconds: float,
        dt: float = 1.0,
        n_paths: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates monte-carlo trajectories of latent prices across the halt window.

        Args:
            p0: Initial observable pre-halt price.
            duration_seconds: Total duration of the halt in seconds.
            dt: Time increment step (default 1.0 second).
            n_paths: Number of Monte Carlo trajectories.

        Returns:
            time_grid: 1D array of time steps [0, duration_seconds].
            paths: 2D array of simulated prices (n_paths, n_steps).
        """
        n_steps = max(int(duration_seconds / dt), 1)
        time_grid = np.linspace(0, duration_seconds, n_steps + 1)
        dt_years = dt / (252.0 * 6.5 * 3600.0)  # Scale seconds to trading years

        # Continuous diffusion component: dW_t
        dw = self.rng.normal(0, np.sqrt(dt_years), size=(n_paths, n_steps))
        continuous_drift = (self.params.drift - 0.5 * self.params.diffusion_vol**2) * dt_years

        # Jump Poisson process: dN_t ~ Poisson(lambda * dt)
        poisson_arrivals = self.rng.poisson(self.params.jump_intensity * dt_years, size=(n_paths, n_steps))

        # Compound jump magnitudes: sum_{i=1}^N J_i
        jump_sizes = np.zeros((n_paths, n_steps))
        arrival_indices = np.where(poisson_arrivals > 0)
        for path_idx, step_idx in zip(*arrival_indices):
            k = poisson_arrivals[path_idx, step_idx]
            jump_sizes[path_idx, step_idx] = np.sum(
                self.rng.normal(self.params.jump_mean, self.params.jump_std, size=k)
            )

        # Log-returns accumulation
        d_log_p = continuous_drift + (self.params.diffusion_vol * dw) + jump_sizes
        log_paths = np.zeros((n_paths, n_steps + 1))
        log_paths[:, 0] = np.log(p0)
        log_paths[:, 1:] = np.log(p0) + np.cumsum(d_log_p, axis=1)

        return time_grid, np.exp(log_paths)


class LiquidityRiskEngine:
    """
    Computes Liquidity-Adjusted Value-at-Risk (L-VaR) and Expected Shortfall (L-CVaR)
    under market depth depletion and exogenous trading halts.
    """

    def __init__(self, alpha: float = 0.99, liquidity_impact_factor: float = 0.5) -> None:
        """
        Args:
            alpha: Tail confidence level (e.g., 0.99 for 99% VaR/CVaR).
            liquidity_impact_factor: Sensitivity eta of spread penalty to market volume.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("Confidence level alpha must be strictly between 0 and 1.")
        self.alpha = alpha
        self.eta = liquidity_impact_factor

    def exogenous_liquidity_cost(
        self,
        position_size: float,
        half_spread: float,
        market_depth: float,
    ) -> float:
        """
        Calculates non-linear liquidation penalty C(V_t):
            C(V_t) = Position * (Half_Spread + eta * (Position / max(V_t, epsilon))^2)
        """
        epsilon = 1e-6
        depth = max(market_depth, epsilon)
        market_impact_premium = self.eta * ((abs(position_size) / depth) ** 2)
        unit_cost = half_spread + market_impact_premium
        return abs(position_size) * unit_cost

    def compute_l_cvar(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        position_size: float,
        half_spreads: np.ndarray,
        market_depths: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculates empirical L-VaR and L-CVaR:
            Loss = -portfolio_value * return + C(V_t)

        Returns:
            (l_var, l_cvar): Tail dollar loss thresholds.
        """
        if len(returns) == 0:
            return 0.0, 0.0

        # Unconstrained standard mark-to-market loss
        pure_pnl_loss = -portfolio_value * returns

        # Exogenous friction penalty vector
        friction_penalties = np.array([
            self.exogenous_liquidity_cost(position_size, s, d)
            for s, d in zip(half_spreads, market_depths)
        ])

        total_loss = pure_pnl_loss + friction_penalties

        # Quantile thresholds
        l_var = float(np.percentile(total_loss, self.alpha * 100))
        tail_losses = total_loss[total_loss >= l_var]

        l_cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else l_var
        return l_var, l_cvar
