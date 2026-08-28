# Experiment Log

This document is intentionally started before optimization.

## E0 — Baseline

**Model:** Logistic Regression  
**Features:** amount + historical behavioral aggregates + categorical context  
**Split:** chronological 70/15/15  
**Threshold:** selected on validation only  
**Test:** untouched until final evaluation

### Hypothesis

A simple, interpretable baseline establishes whether the synthetic world contains learnable signal before adding graph features or an LLM.

### What we will record

- precision
- recall
- F1
- false-positive rate
- false-positive cost
- false-negative cost
- calibration
- threshold
- inference latency

Never replace failed numbers with hand-picked examples.
