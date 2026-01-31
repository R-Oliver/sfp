from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import jax


@dataclass
class NumericsResult:
    passed: bool
    max_diff: float
    mean_diff: float
    median_diff: float
    pct_above_threshold: float
    threshold: float
    worst_idx: tuple[int, ...]
    worst_ref_val: float
    worst_test_val: float
    shape: tuple[int, ...]
    # Region analysis for 2D arrays
    region_stats: Optional[dict[tuple[int, int], dict[str, float]]] = None
    region_grid: Optional[tuple[int, int]] = None

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"NumericsResult({status})",
            f"  shape:     {self.shape}",
            f"  max_diff:  {self.max_diff:.6f}",
            f"  mean_diff: {self.mean_diff:.6f}",
            f"  median:    {self.median_diff:.6f}",
            f"  % > {self.threshold}: {self.pct_above_threshold:.2f}%",
            f"  worst at {self.worst_idx}: ref={self.worst_ref_val:.4f}, test={self.worst_test_val:.4f}",
        ]
        if self.region_stats and self.region_grid:
            lines.append(f"  regions ({self.region_grid[0]}x{self.region_grid[1]}):")
            for (i, j), stats in sorted(self.region_stats.items()):
                lines.append(f"    [{i},{j}]: mean={stats['mean']:.4f}, %bad={stats['pct_bad']:.1f}%")
        return "\n".join(lines)


def compare(
    ref: jax.Array,
    test: jax.Array,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    threshold: float = 0.1,
    region_grid: Optional[tuple[int, int]] = None,
) -> NumericsResult:
    """
    Compare two arrays and return detailed numerics analysis.

    Args:
        ref: Reference array (ground truth)
        test: Test array to validate
        rtol: Relative tolerance for pass/fail
        atol: Absolute tolerance for pass/fail
        threshold: Absolute threshold for "bad" element counting
        region_grid: If provided, analyze errors by region. E.g., (2, 2) splits
            into 4 regions matching a 2x2 mesh, (4, 4) for a 4x4 mesh, etc.
            Only applies to 2D arrays.

    Returns:
        NumericsResult with detailed comparison statistics
    """
    passed = bool(jnp.allclose(ref, test, rtol=rtol, atol=atol))
    diff = jnp.abs(ref - test)

    max_diff = float(jnp.max(diff))
    mean_diff = float(jnp.mean(diff))
    median_diff = float(jnp.median(diff))
    pct_above = float(100 * jnp.mean(diff > threshold))

    # Find worst error location
    flat_idx = int(jnp.argmax(diff))
    worst_idx = tuple(int(i) for i in jnp.unravel_index(flat_idx, diff.shape))
    worst_ref = float(ref[worst_idx])
    worst_test = float(test[worst_idx])

    # Region analysis for 2D arrays
    region_stats = None
    if region_grid is not None and ref.ndim == 2:
        gh, gw = region_grid
        h, w = ref.shape
        region_h, region_w = h // gh, w // gw
        region_stats = {}
        for i in range(gh):
            for j in range(gw):
                region = diff[
                    i * region_h : (i + 1) * region_h,
                    j * region_w : (j + 1) * region_w,
                ]
                region_stats[(i, j)] = {
                    "mean": float(jnp.mean(region)),
                    "pct_bad": float(100 * jnp.mean(region > threshold)),
                }

    return NumericsResult(
        passed=passed,
        max_diff=max_diff,
        mean_diff=mean_diff,
        median_diff=median_diff,
        pct_above_threshold=pct_above,
        threshold=threshold,
        worst_idx=worst_idx,
        worst_ref_val=worst_ref,
        worst_test_val=worst_test,
        shape=ref.shape,
        region_stats=region_stats,
        region_grid=region_grid,
    )
