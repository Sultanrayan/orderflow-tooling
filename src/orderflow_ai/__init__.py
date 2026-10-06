"""OrderFlow Engineer: AI-powered order flow analysis.

The package is organised as a five layer pipeline. Each layer lives in its own
sub-package and only depends on the layers below it:

``orderflow_ai.ingestion``
    Layer 1 - exchange feeds, normalisation, buffering, timestamp alignment.
``orderflow_ai.primitives``
    Layer 2 - the ten order flow primitives.
``orderflow_ai.patterns``
    Layer 3 - discrete pattern detection and fallback scoring.
``orderflow_ai.context``
    Layer 4 - market structure, key levels, sessions, volatility.
``orderflow_ai.output``
    Layer 5 - summarisation, prioritisation and token optimised payloads.
``orderflow_ai.ai``
    Provider clients plus strict validation of the model response.
``orderflow_ai.core``
    Shared models, configuration and the pipeline orchestrator.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"