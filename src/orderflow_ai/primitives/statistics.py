"""Small statistics helpers shared by the primitives.

Kept dependency free and explicit about behaviour on empty input so callers
never have to guard against division by zero themselves.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "clamp",
    "linear_slope",
    "mean",
    "percentile_rank",
    "safe_div",
    "stdev",
]


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide ``numerator`` by ``denominator``, returning ``default`` on zero."""
    return numerator / denominator if denominator else default


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Constrain ``value`` to the inclusive range ``[minimum, maximum]``."""
    return max(minimum, min(maximum, value))


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, ``0.0`` for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    """Population standard deviation, ``0.0`` for fewer than two values."""
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return variance**0.5


def percentile_rank(value: float, sample: Sequence[float]) -> float:
    """Average rank of ``value`` within ``sample``, as a fraction.

    Tied entries occupy a block of positions; ``value`` is placed at the middle
    of that block, so ``percentile_rank(2.0, [1.0, 2.0, 3.0, 4.0])`` is ``0.5``
    and a value above every entry in the sample reaches ``1.0``. A perfectly
    flat series has no rank to distinguish it, so it reports ``0.5`` instead of
    the misleading ``1.0`` a plain "less than or equal" count would produce.

    Returns:
        A value in ``[0, 1]``. An empty sample yields ``0.0``.
    """
    if not sample:
        return 0.0
    below = sum(1 for candidate in sample if candidate < value)
    equal = sum(1 for candidate in sample if candidate == value)
    size = len(sample)
    if equal == size:
        return 0.5
    if equal == 0:
        return below / size
    return (below + (equal + 1) / 2) / size


def linear_slope(values: Sequence[float]) -> float:
    """Least squares slope of ``values`` against their index.

    Returns:
        Slope in output units per step, ``0.0`` when fewer than two points exist
        or when every point is identical.
    """
    count = len(values)
    if count < 2:
        return 0.0
    average = mean(values)
    index_average = (count - 1) / 2
    covariance = sum((index - index_average) * (value - average) for index, value in enumerate(values))
    denominator = sum((index - index_average) ** 2 for index in range(count))
    return safe_div(covariance, denominator)