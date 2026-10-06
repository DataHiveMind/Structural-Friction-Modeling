"""
src/causal/causal_engine.py

Causal inference engine utilizing DoWhy for structural halt effect estimation.
Controls for exogenous macro-confounders X using Pearl's backdoor criterion
to isolate structural liquidity shock penalties from fundamental repricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import dowhy
from dowhy import CausalModel
import pandas as pd


@dataclass
class CausalEstimationResult:
    """Summary container for treatment effect metrics and refutation checks."""
    ate: float
    estimate_method: str
    identified_estimand: Any
    refutation_results: Dict[str, Any]
    summary_text: str


class HaltCausalEngine:
    """
    Constructs causal DAGs, identifies backdoor adjustment sets, and isolates
    the Average Treatment Effect (ATE) of exchange halts on market quality metrics.
    """

    def __init__(
        self,
        treatment_col: str = "treatment_group",
        outcome_col: str = "effective_spread",
        confounder_cols: Optional[List[str]] = None,
    ) -> None:
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.confounder_cols = confounder_cols or [
            "benchmark_return",
            "vix_level",
            "etf_net_flow",
            "realized_volatility",
        ]
        self.model: Optional[CausalModel] = None

    def _build_dot_graph(self) -> str:
        """
        Constructs the causal Directed Acyclic Graph (DAG) specification in DOT syntax.
        Confounders X point to Treatment T and Outcome Y. Latent price discovery
        and microstructural frictions are disentangled.
        """
        confounder_str = " ".join([f"{c} -> {self.treatment_col}; {c} -> {self.outcome_col};" for c in self.confounder_cols])
        dot_graph = f"""
        digraph {{
            {confounder_str}
            {self.treatment_col} -> {self.outcome_col};
        }}
        """
        return dot_graph.strip()

    def fit_and_estimate(
        self,
        data: pd.DataFrame,
        estimation_method: str = "backdoor.linear_regression",
    ) -> CausalEstimationResult:
        """
        Initializes the structural graph, executes backdoor criterion identification,
        and computes the Average Treatment Effect (ATE).

        Args:
            data: Time-aligned DataFrame containing treatment, outcome, and confounders.
            estimation_method: DoWhy estimation algorithm identifier.

        Returns:
            CausalEstimationResult holding ATE and diagnostic details.
        """
        clean_df = data[[self.treatment_col, self.outcome_col] + self.confounder_cols].dropna().copy()
        dot_dag = self._build_dot_graph()

        self.model = CausalModel(
            data=clean_df,
            treatment=self.treatment_col,
            outcome=self.outcome_col,
            graph=dot_dag,
        )

        # Step 1: Formal identification via backdoor adjustment
        identified_estimand = self.model.identify_effect(proceed_when_unidentifiable=True)

        # Step 2: Estimate the target effect E[Y | do(T=1)] - E[Y | do(T=0)]
        estimate = self.model.estimate_effect(
            identified_estimand,
            method_name=estimation_method,
            test_significance=True,
        )

        ate_value = float(estimate.value)

        # Step 3: Run refutation stress tests
        refutations = self._run_refutations(identified_estimand, estimate)

        summary = (
            f"Causal Estimation Completed [{estimation_method}]\n"
            f"Average Treatment Effect (ATE) on {self.outcome_col}: {ate_value:.6f}\n"
            f"Controlled Confounders: {', '.join(self.confounder_cols)}\n"
            f"Refutation checks passed: {all(r.get('passed', False) for r in refutations.values())}"
        )

        return CausalEstimationResult(
            ate=ate_value,
            estimate_method=estimation_method,
            identified_estimand=identified_estimand,
            refutation_results=refutations,
            summary_text=summary,
        )

    def _run_refutations(
        self, identified_estimand: Any, estimate: Any
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executes robustness refutations:
        1. Placebo Treatment Refuter: Replaces T with random noise; expected effect -> 0.
        2. Random Common Cause: Injects an independent confounder; estimate should remain stable.
        3. Data Subset Refuter: Assesses sensitivity across 80% bootstrap sub-samples.
        """
        refutation_results = {}

        if self.model is None:
            return refutation_results

        # 1. Placebo Treatment
        try:
            placebo = self.model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
            )
            refutation_results["placebo_treatment"] = {
                "new_effect": float(placebo.new_effect),
                "passed": abs(placebo.new_effect) < abs(estimate.value) * 0.25,
            }
        except Exception as e:
            refutation_results["placebo_treatment"] = {"error": str(e), "passed": False}

        # 2. Random Common Cause
        try:
            random_cause = self.model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause",
            )
            diff = abs(random_cause.new_effect - estimate.value) / (abs(estimate.value) + 1e-8)
            refutation_results["random_common_cause"] = {
                "new_effect": float(random_cause.new_effect),
                "passed": diff < 0.15,
            }
        except Exception as e:
            refutation_results["random_common_cause"] = {"error": str(e), "passed": False}

        # 3. Data Subset Refuter
        try:
            subset = self.model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="data_subset_refuter",
                subset_fraction=0.8,
            )
            diff = abs(subset.new_effect - estimate.value) / (abs(estimate.value) + 1e-8)
            refutation_results["data_subset"] = {
                "new_effect": float(subset.new_effect),
                "passed": diff < 0.20,
            }
        except Exception as e:
            refutation_results["data_subset"] = {"error": str(e), "passed": False}

        return refutation_results
