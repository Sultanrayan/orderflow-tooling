# OrderFlow Engineer

An AI powered order flow analysis engine that transforms raw market microstructure data into structured, AI-ready signals.

The system collects tick level data and order book depth from exchanges, computes order flow primitives, detects patterns, builds market context, and produces a compact JSON payload optimized for Large Language Models. The AI returns a trading signal with confidence, setup, and invalidation levels.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Layer Breakdown](#layer-breakdown)
4. [Project Structure](#project-structure)
5. [Data Flow](#data-flow)
6. [Order Flow Primitives](#order-flow-primitives)
7. [Patterns](#patterns)
8. [Token Optimization](#token-optimization)
9. [AI Integration](#ai-integration)
10. [Installation](#installation)
11. [Configuration](#configuration)
12. [Usage](#usage)
13. [Testing](#testing)
14. [Deployment](#deployment)
15. [Roadmap](#roadmap)
16. [License](#license)

---

## Overview

OrderFlow AI is a modular pipeline designed for traders, quantitative researchers, and developers who want to feed structured order flow data into AI models for market analysis.

### Key Principles

- **AI-first output**: Data is formatted for AI consumption, not human charts.
- **No forced signals**: `NO_TRADE` and `WAIT` are valid outcomes.
- **Confidence-driven**: Every signal carries a quantified confidence score.
- **Fallback reasoning**: When no pattern is detected, confidence is computed from primitives.
- **Token efficient**: Payloads are compressed up to 96 percent compared to raw JSON.

### What It Does

- Collects real-time market data via WebSocket and REST.
- Computes 10 core order flow primitives.
- Detects patterns such as absorption, imbalance, and trapped traders.
- Builds multi-timeframe context and key levels.
- Produces a compact JSON payload for an LLM.
- Validates the AI response against a strict schema.

---

## Architecture

```mermaid
graph TD
    A[Exchange API] --> B[Layer 1: Data Ingestion]
    B --> C[Layer 2: Order Flow Primitives]
    C --> D[Layer 3: Order Flow Patterns]
    D --> E[Layer 4: Context and Structure]
    E --> F[Layer 5: AI-Ready Output]
    F --> G[AI Model API]
    G --> H[Signal Validation]
    H --> I[Trading Signal]
```

### Layer Responsibilities

```mermaid
graph LR
    L1[Layer 1<br/>Ingestion] --> L2[Layer 2<br/>Primitives]
    L2 --> L3[Layer 3<br/>Patterns]
    L2 --> L4[Layer 4<br/>Context]
    L3 --> L5[Layer 5<br/>AI Output]
    L4 --> L5
    L5 --> AI[AI Model]
```

---

## Layer Breakdown

### Layer 1: Data Ingestion

Collects raw market data from exchange feeds.

| Component | Responsibility |
|-----------|---------------|
| TradeFeed | WebSocket stream of executed trades |
| DepthFeed | WebSocket stream of order book updates |
| OHLCVFeed | REST polling for candle data |
| Normalizer | Standardizes format across exchanges |
| Buffer | Ring buffer for tick retention |
| TimestampSync | Aligns timestamps across sources |

### Layer 2: Order Flow Primitives

Computes 10 high-quality primitives grouped into four categories.

```mermaid
graph TD
    P[Primitives] --> S[Structure]
    P --> D[Direction]
    P --> PR[Pressure]
    P --> Q[Quality]
    
    S --> S1[Footprint]
    S --> S2[Volume Profile]
    S --> S3[Value Migration]
    
    D --> D1[Delta]
    D --> D2[CVD]
    D --> D3[Delta Divergence]
    
    PR --> P1[Imbalance]
    PR --> P2[Orderbook Depth]
    
    Q --> Q1[Volume Intensity]
    Q --> Q2[Absorption Score]
```

### Layer 3: Order Flow Patterns

Detects discrete patterns with confidence scores.

| Pattern | Description |
|---------|-------------|
| Absorption | Large volume absorbed without price movement |
| Trapped Traders | Failed breakout with rapid reversal |
| Liquidity Sweep | Stop hunt above or below key level |
| Delta Divergence | Price and delta move in opposite directions |
| Exhaustion | Trend momentum fading |
| Iceberg | Hidden large orders revealed by repeated fills |
| Unfinished Auction | Price likely to return to fill gap |

### Layer 4: Context and Structure

Provides market context.

- Market structure: HH, HL, LH, LL, BOS, CHoCH
- Key levels: support, resistance, prior POC
- Session: Asia, London, New York
- Multi-timeframe alignment
- Volatility regime

### Layer 5: AI-Ready Output

Transforms all data into a compact JSON payload.

```mermaid
graph LR
    A[Primitives] --> S[Summarizer]
    B[Patterns] --> S
    C[Context] --> S
    S --> P[Prioritizer]
    P --> F[Formatter]
    F --> O[Optimized JSON]
    O --> AI[AI Model]
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant EX as Exchange
    participant L1 as Layer 1 Ingestion
    participant L2 as Layer 2 Primitives
    participant L3 as Layer 3 Patterns
    participant L4 as Layer 4 Context
    participant L5 as Layer 5 Output
    participant AI as AI Model
    
    EX->>L1: Ticks and Orderbook
    L1->>L2: Structured data
    L2->>L3: 10 Primitives
    L2->>L4: 10 Primitives
    L3->>L5: Patterns
    L4->>L5: Context
    L5->>AI: Optimized JSON
    AI->>L5: Signal
    L5->>L5: Validate
```

### Timing

| Stage | Duration |
|-------|----------|
| Layer 1 Ingestion | 50 ms |
| Layer 2 Primitives | 12 ms |
| Layer 3 Patterns | 8 ms |
| Layer 4 Context | 5 ms |
| Layer 5 Output | 3 ms |
| AI Model | 2000 ms |
| Validation | 2 ms |
| Total | 2080 ms |

---

## Order Flow Primitives

Ten primitives organized into four groups.

### Structure

| Primitive | Description | Output |
|-----------|-------------|--------|
| Footprint | Price by bid and ask volume | OHLCV plus level map |
| Volume Profile | Volume distribution | POC, VAH, VAL, HVN, LVN |
| Value Migration | POC and VA shift over time | Direction and strength |

### Direction

| Primitive | Description | Output |
|-----------|-------------|--------|
| Delta | Buy minus sell aggressor volume | Value, ratio, strength |
| CVD | Cumulative delta | Value, slope, momentum |
| Delta Divergence | Price versus delta mismatch | Type and confidence |

### Pressure

| Primitive | Description | Output |
|-----------|-------------|--------|
| Imbalance | Bid and ask ratio per level | List and strongest |
| Orderbook Depth | Live book metrics | Depth, imbalance, walls |

### Quality

| Primitive | Description | Output |
|-----------|-------------|--------|
| Volume Intensity | Volume relative to average | Ratio and percentile |
| Absorption Score | Volume absorbed without price move | Confidence score |

### Primitive Output Example

```json
{
  "structure": {
    "footprint": {
      "open": 1.0849,
      "high": 1.0855,
      "low": 1.0849,
      "close": 1.0854,
      "total_volume": 120.0,
      "levels": [
        {"price": 1.0852, "bid": 20.0, "ask": 22.0}
      ]
    },
    "volume_profile": {
      "poc": 1.0852,
      "vah": 1.0854,
      "val": 1.0851
    },
    "value_migration": {
      "poc_direction": "up",
      "migration_strength": 0.65
    }
  },
  "direction": {
    "delta": {"value": 8.5, "ratio": 0.071, "direction": "bullish"},
    "cvd": {"value": 142.3, "slope": 0.85, "momentum": "accelerating"},
    "delta_divergence": {"detected": false}
  },
  "pressure": {
    "imbalance": {"strongest": {"price": 1.0853, "type": "buy", "ratio": 1.25}},
    "orderbook": {"imbalance_ratio": 1.35, "imbalance_direction": "bid"}
  },
  "quality": {
    "volume_intensity": {"ratio": 1.33, "intensity": "high"},
    "absorption": {"detected": true, "confidence": 78}
  }
}
```

---

## Patterns

Patterns are computed with confidence scores between 0 and 100.

### Confidence Tiers

```mermaid
graph LR
    A[Pattern Confidence] --> B[75-100: HIGH]
    A --> C[50-74: MEDIUM]
    A --> D[0-49: LOW]
    B --> E[Full position]
    C --> F[Small position or WAIT]
    D --> G[NO TRADE]
```

### Detection Rules

| Pattern | Rule | Confidence Inputs |
|---------|------|-------------------|
| Absorption | High volume, low price movement, high delta | Volume, delta, range, duration |
| Trapped Traders | Failed breakout plus rapid reversal | Breakout strength, reversal speed |
| Liquidity Sweep | Brief overshoot above or below key level | Level proximity, reversal speed |
| Delta Divergence | Price and delta move in opposite directions | Bars observed, magnitude |
| Exhaustion | Declining volume and delta at trend end | Trend length, volume decay |
| Iceberg | Repeated fills at same level with hidden size | Fill pattern, level persistence |
| Unfinished Auction | One side zero at extreme | Volume gap, price location |

### No-Pattern Fallback

When no pattern is detected, confidence is computed from primitives.

```mermaid
graph TD
    A[Check Patterns] --> B{Pattern found?}
    B -->|Yes| C[Use pattern confidence]
    B -->|No| D[Use fallback engine]
    D --> E[Delta strength]
    D --> F[Volume intensity]
    D --> G[Orderbook imbalance]
    D --> H[MTF alignment]
    D --> I[Key level proximity]
    D --> J[Volatility regime]
    E --> K[Sum scores]
    F --> K
    G --> K
    H --> K
    I --> K
    J --> K
    K --> L[Classify HIGH, MEDIUM, LOW, NONE]
```

### Fallback Scoring Weights

| Component | Max Points |
|-----------|------------|
| Delta strength | 20 |
| Orderbook imbalance | 20 |
| MTF alignment | 20 |
| Volume intensity | 15 |
| Key level proximity | 15 |
| Volatility regime | 10 |
| Total | 100 |

---

## Token Optimization

Payload size is reduced through ten strategies.

```mermaid
graph LR
    A[Raw JSON 5.5 KB] --> B[Compact]
    B --> C[Array Encoding]
    C --> D[Short Keys]
    D --> E[Remove Nulls]
    E --> F[Round Numbers]
    F --> G[Delta Encode]
    G --> H[Summary]
    H --> I[Cache Static]
    I --> J[Tiered]
    J --> K[Final 0.2 KB]
```

### Strategy Comparison

| Strategy | Reduction | Complexity |
|----------|-----------|------------|
| Compact JSON | 60 percent | Low |
| Array Encoding | 40 percent | Low |
| Short Keys | 30 percent | Low |
| Remove Nulls | 15 percent | Low |
| Round Numbers | 10 percent | Low |
| Delta Encoding | 50 percent | Medium |
| Summary Instead of Raw | 80 percent | Medium |
| Cache Static Parts | 90 percent | High |
| Tiered Complexity | 85 percent | High |

### Tiered Prompt Selection

| Tier | Trigger | Tokens |
|------|---------|--------|
| Minimal | Two or more patterns, confidence above 80 | 150 |
| Standard | Normal conditions | 500 |
| Full | No patterns or confidence below 50 | 1500 |

### Cost Impact

| Version | Tokens per Call | Monthly Calls | Monthly Cost |
|---------|-----------------|---------------|--------------|
| Full | 2300 | 43200 | 498 USD |
| Compact | 900 | 43200 | 195 USD |
| Optimized | 300 | 43200 | 65 USD |
| Tiered | 150 | 43200 | 33 USD |
| Cached | 150 | 43200 | 15 USD |

---

## AI Integration

### Supported Providers

- OpenAI
- Anthropic
- Ollama

### Payload Example

```json
{
  "model": "gpt-4o",
  "temperature": 0.2,
  "max_tokens": 300,
  "response_format": {"type": "json_object"},
  "messages": [
    {
      "role": "system",
      "content": "Order flow analyst. Protect capital. NO_TRADE valid. Never invent patterns."
    },
    {
      "role": "user",
      "content": "{\"m\":{\"s\":\"BTCUSDT\",\"tf\":\"1m\"},\"d\":{\"v\":8.5,\"dir\":\"bull\"},\"pat\":[[\"absorption\",1.0850,\"bull\",78]]}"
    }
  ]
}
```

### Response Schema

```json
{
  "bias": "bullish",
  "confidence": 78,
  "signal": "BUY",
  "primary_evidence": "patterns",
  "reasoning": "Strong absorption at 1.0850.",
  "key_levels": {
    "support": [1.0852, 1.0850],
    "resistance": [1.0855, 1.0858]
  },
  "setup": {
    "entry": 1.0853,
    "stop": 1.0849,
    "target": 1.0862,
    "rr": 2.25
  },
  "wait_for": null,
  "invalidation": "Close below 1.0850",
  "risk_notes": "R:R 2.25 acceptable."
}
```

### Response Validation

```mermaid
graph TD
    A[AI Response] --> B{Schema valid?}
    B -->|No| C[REJECT]
    B -->|Yes| D{Confidence matches signal?}
    D -->|No| C
    D -->|Yes| E{R:R above 1.5?}
    E -->|No| C
    E -->|Yes| F{Patterns hallucinated?}
    F -->|Yes| C
    F -->|No| G[ACCEPT]
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Redis (for caching)
- PostgreSQL (for historical data)
- An AI API key (OpenAI, Anthropic, or local Ollama)

### Setup

```bash
git clone https://github.com/your-org/orderflow-ai.git
cd orderflow-ai
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...
BINANCE_API_KEY=...
BINANCE_SECRET=...
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/orderflow
```

---

## Configuration

`configs/default.yaml`:

```yaml
exchange:
  name: binance
  symbol: BTCUSDT
  tick_size: 1.0
  depth_levels: 20

pipeline:
  timeframe: 1m
  buffer_size: 10000
  history_bars: 20

primitives:
  delta_ratio_threshold: 0.15
  imbalance_ratio_threshold: 1.5
  absorption_min_confidence: 60

patterns:
  min_confidence: 50
  max_patterns: 5
  max_age_bars: 3

ai:
  provider: openai
  model: gpt-4o
  temperature: 0.2
  max_tokens: 300

optimization:
  tier: auto
  compact_json: true
  short_keys: true
  delta_encode: true
```

---

## Usage

### Live Analysis

```bash
python scripts/live_run.py --config configs/binance.yaml
```

### Backtest

```bash
python scripts/backtest.py --start 2024-01-01 --end 2024-10-01
```

### Collect Data

```bash
python scripts/collect_data.py --symbol BTCUSDT --duration 3600
```

### Python API

```python
import asyncio
from orderflow_ai.core.pipeline import OrderFlowPipeline
from orderflow_ai.ai.client import AIClient

async def main():
    pipeline = OrderFlowPipeline(config="configs/binance.yaml")
    ai = AIClient(provider="openai")

    async for analysis in pipeline.stream():
        payload = pipeline.build_payload(analysis)
        signal = await ai.analyze(payload)
        print(signal)

asyncio.run(main())
```

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

| Test File | Coverage |
|-----------|----------|
| test_layer1.py | Data ingestion and buffering |
| test_layer2.py | Primitive computation |
| test_layer3.py | Pattern detection |
| test_layer4.py | Context building |
| test_layer5.py | Output formatting and validation |

---



### Production Checklist

- Enable Redis caching for primitives
- Set up monitoring with Prometheus and Grafana
- Configure log aggregation
- Use environment-specific config files
- Enable API rate limiting
- Set up alerting for signal output
- Implement database backups

---

## Roadmap

```mermaid
gantt
    title OrderFlow AI Roadmap
    dateFormat YYYY-MM-DD
    section Phase 1
    Data ingestion           :a1, 2026-01-01, 14d
    Primitives               :a2, after a1, 21d
    section Phase 2
    Pattern detection        :b1, after a2, 21d
    Context building         :b2, after b1, 14d
    section Phase 3
    AI output layer          :c1, after b2, 14d
    Token optimization       :c2, after c1, 10d
    section Phase 4
    Backtest framework       :d1, after c2, 21d
    Live deployment          :d2, after d1, 14d
```

### Future Features

- Multi-exchange aggregation
- Additional primitive support for futures and options
- Fine-tuned local model
- Web dashboard for monitoring
- Alert integrations (Telegram, Discord, Slack)
- Feedback loop for prompt improvement

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Push to the branch
5. Open a pull request

Please follow the existing code style and include tests for new features.

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

For questions or support, open an issue on GitHub or contact the maintainers.

---

## Acknowledgments

Built with principles from order flow trading, market microstructure research, and modern AI prompt engineering.