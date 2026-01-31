import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class BenchmarkResult:
    mean_ms: float
    median_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    n_iters: int
    n_warmup: int
    times_ms: list[float]

    def __repr__(self) -> str:
        lines = [
            f"BenchmarkResult ({self.n_iters} iters, {self.n_warmup} warmup)",
            f"  mean:   {self.mean_ms:>10.3f} ms",
            f"  median: {self.median_ms:>10.3f} ms",
            f"  stdev:  {self.stdev_ms:>10.3f} ms",
            f"  min:    {self.min_ms:>10.3f} ms",
            f"  max:    {self.max_ms:>10.3f} ms",
            f"  p95:    {self.p95_ms:>10.3f} ms",
            f"  p99:    {self.p99_ms:>10.3f} ms",
        ]
        return "\n".join(lines)


def benchmark(
    fn: Callable[..., Any],
    *args,
    n_iters: int = 50,
    n_warmup: int = 3,
    **kwargs,
) -> BenchmarkResult:
    """
    Benchmark a JAX function with proper synchronization.

    This handles the async dispatch correctly by calling block_until_ready()
    on the result before and after timing.

    Args:
        fn: Function to benchmark (should return a JAX array)
        *args: Positional arguments to pass to fn
        n_iters: Number of timed iterations
        n_warmup: Number of warmup iterations (not timed)
        **kwargs: Keyword arguments to pass to fn

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup: compile and run a few times
    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
        result.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_iters):
        result.block_until_ready()

        start = time.perf_counter()
        result = fn(*args, **kwargs)
        result.block_until_ready()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # ms

    return BenchmarkResult(
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times),
        max_ms=max(times),
        p95_ms=float(np.percentile(times, 95)),
        p99_ms=float(np.percentile(times, 99)),
        n_iters=n_iters,
        n_warmup=n_warmup,
        times_ms=times,
    )
