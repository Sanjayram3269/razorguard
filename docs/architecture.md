# RazorGuard Architecture — v0.1

The current implementation deliberately starts with the evaluation foundation.

## Decision pipeline

```text
Historical transaction prefix
          |
          v
Leakage-safe feature engine
          |
          v
Baseline risk model
          |
          v
Validation threshold
          |
          v
Held-out chronological test
```

## Future production-style pipeline

```text
                         +----------------------+
                         | Transaction Gateway  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Historical Features  |
                         +----------+-----------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
             +-------------+                +---------------+
             | Risk Model  |                | Entity Graph  |
             +------+------+                +-------+-------+
                    |                               |
                    +---------------+---------------+
                                    v
                           +----------------+
                           | Risk Decision  |
                           +-------+--------+
                                   |
                         high-risk / review
                                   v
                        +--------------------+
                        | AI Investigator    |
                        | evidence tools     |
                        +---------+----------+
                                  |
                                  v
                        +--------------------+
                        | Policy Gate        |
                        | bounded actions    |
                        +---------+----------+
                                  |
                                  v
                           Human Review
                                  |
                                  v
                        +--------------------+
                        | Audit Event Store  |
                        +--------------------+
```

## Non-negotiable safety rule

The LLM cannot independently change the risk score or authorize a consequential financial action. It produces a structured investigative case from bounded evidence. The policy layer controls what may happen next.
