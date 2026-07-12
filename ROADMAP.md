# Tier-3 Roadmap — Named, Not Hand-Waved

The one-slide articulation required by plan.md §12.10 and Definition-of-Done item #6 (§19.4).
Everything below is **deliberately not built**. Each row names the real library and the honest
reason it is out of scope — framed as "what we'd do with more time," never as "half-built."

| Roadmap item | Why it's out of scope now | Concrete next step |
|---|---|---|
| Full trained GNN (GAT/GraphSAGE) for supply-chain contagion | Needs real relational data at scale (GST e-way-bill graphs) + multi-epoch GPU training; training on our synthetic graph would look trained while having learned nothing | `PyTorch Geometric` on real e-way-bill graphs once sandbox access is granted (upgrade path from the graph-lite features shipped in Phase 5) |
| Neural survival models (DeepSurv, Cox-Time, DeepHit) | Marginal uplift over the discrete-time LightGBM hazard model doesn't justify the added training/validation complexity in a time-boxed build | `pycox`, once a larger longitudinal panel (Freddie-Mac-scale or the real sandbox panel) is available |
| Federated learning across lenders | Requires multiple participating institutions' infrastructure — structurally unavailable to a single team | `Flower` or `TensorFlow Federated`, aligned with FREE-AI's data-infrastructure recommendations |
| Causal-uplift reasoning (why risk rose, not just what correlates) | Needs experimental variation or substantially more data to estimate credible causal effects | `DoWhy` / `EconML` layered on top of the existing SHAP output |
| Cross-jurisdiction domain adaptation | Out of scope for an India-specific, RBI-aligned build | Future-markets roadmap slide only |
| Conformal-prediction-grade uncertainty intervals | The calibration-residual confidence bands shipped in Phase 7 already cover the "flag for human review" use case at a fraction of the cost | `MAPIE` or `crepes` |
| Learned cross-attention multimodal fusion | Requires a custom PyTorch model + training-iteration time; the shipped intermediate fusion (dense graph/text vectors concatenated before the LightGBM head, Phase 5) already avoids scalar compression | Small cross-attention fusion head (plan.md §9.5 diagram) once real text/graph modalities exist |
| Real Feast + Postgres + Redpanda deployment | No Docker daemon in the build environment; local stand-ins keep identical contracts | `docker compose up` — the compose file, env-var switches (`DATABASE_URL`, `KAFKA_BOOTSTRAP_SERVERS`), and Kafka-wire-compatible bus abstraction are already wired |

## Honest data caveats to state alongside the roadmap

- Graph and text features are **synthetic, provenance-tagged** (`synthetic_graph` / `synthetic_text`), generated from observable behavior only (never the label). The Phase 5 ablation therefore shows ~zero uplift (fused AUC 0.7145 vs 0.7142) — reported verbatim as evidence of methodological honesty; the mechanism activates with real GST/UPI counterparty data.
- The Home Credit proxy has no region/gender-of-promoter fields, so those fairness slices are shown as "activates with sandbox data" rather than faked.
- Raw monthly installment data was unavailable at Tier-2 build time, so the fused ablation runs at loan level; the committed person-period hazard model remains the core 12-month engine.
