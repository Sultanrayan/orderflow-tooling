"""Detector interface.

A detector turns the analysed bar into either a :class:`PatternMatch` or
``None``. Detectors never raise for "not applicable": an unmet rule simply means
no pattern, which is a valid and frequent outcome.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .models import PatternContext, PatternMatch

__all__ = ["PatternDetector"]


class PatternDetector(ABC):
    """Base class for every pattern detector."""

    name: ClassVar[str] = "pattern"
    label: ClassVar[str] = "Pattern"

    @abstractmethod
    def detect(self, context: PatternContext) -> PatternMatch | None:
        """Return a match when the pattern is present, otherwise ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{type(self).__name__}(name={self.name!r})"