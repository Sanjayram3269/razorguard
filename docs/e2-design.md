# RazorGuard E2 — Realistic Behavioral Risk World

## Objective

Create a synthetic payment ecosystem where chargeback risk emerges from
behavioral patterns, account state, transaction velocity, and relationships
between entities.

The objective is not to maximize model performance artificially.

The objective is to create a realistic experimental environment where the
model must generalize from observable behavior to future outcomes.

## Behavioral Archetypes

### 1. Normal

Stable purchasing behavior with low chargeback probability.

### 2. New Account

Recently created account with limited history.

New accounts are not automatically risky.

### 3. High-Value Legitimate

Long-lived customers capable of large legitimate transactions.

Large transaction amount alone must not imply fraud.

### 4. Compromised Account

An account with stable historical behavior followed by an abrupt behavioral
shift.

Examples:

- sudden increase in transaction amount
- unusual transaction velocity
- location change
- new device
- unusual payment behavior

### 5. Burst Behavior

A short period containing unusually dense transaction activity.

Burst behavior increases risk but is not automatically fraudulent.

### 6. Coordinated Cluster

Multiple accounts interacting through shared infrastructure such as:

- devices
- IP/network identifiers
- payment instruments

The relationship itself is evidence, not proof of fraud.

### 7. Legitimate Shared Infrastructure

Multiple legitimate accounts may share infrastructure.

This prevents the model from learning the incorrect rule:

    shared device = fraud

## Temporal Design

Transactions are generated chronologically.

The dataset is split into:

- 70% training
- 15% validation
- 15% future test

The test period occurs strictly after the training period.

## Temporal Drift

The future period contains moderate behavioral changes.

Examples:

- different merchant mix
- changed transaction amounts
- changed burst frequency
- changed geographic distribution

The model should therefore be evaluated on future behavior rather than
randomly sampled rows.

## Label Design

Chargeback is a future outcome.

Runtime features must never use:

- chargeback outcome
- chargeback timestamp
- future dispute information

## Evaluation

Primary metrics:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC
- False Positive Rate

Decision metrics:

- False Positive Cost
- False Negative Cost
- Expected Error Cost

Threshold selection occurs exclusively on the validation set.

The test set remains untouched until final evaluation.

## E2 Success Criteria

E2 is successful only if:

1. The generated data contains the intended behavioral archetypes.
2. Unit tests verify the behavioral properties.
3. Historical features remain leakage-safe.
4. Future test data is chronologically separated.
5. Model performance is measured on unseen future behavior.
6. Improvements are reported honestly, including regressions.