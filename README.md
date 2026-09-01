# RazorGuard

**Evidence-grounded fraud-risk investigation platform.**

RazorGuard is a production-style fraud investigation system that combines deterministic risk scoring, network intelligence, coordinated-risk detection, and AI-assisted analysis to help investigators understand and act on transaction risk.

---

## Problem

Transaction-level fraud scores alone do not explain **coordinated** or **relational** risk. A single blocked transaction may be part of a larger fraud ring involving shared devices, merchant concentration, or temporal bursts. RazorGuard surfaces this intelligence.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND (React)                       │
│  Command Center · Case Queue · Investigation · Analytics    │
└────────────────────────────┬────────────────────────────────┘
                             │ API
┌────────────────────────────▼────────────────────────────────┐
│                     FastAPI Backend                          │
├─────────────────────────────────────────────────────────────┤
│  Transaction Scoring    │  Case Management                  │
│  Risk Fusion            │  Investigation Workflow           │
│  Behavioral Features    │  Audit Trail                      │
├─────────────────────────────────────────────────────────────┤
│  Risk Engine                                                │
│  ├── ML Model (Logistic Regression)                         │
│  ├── Network Intelligence (Entity Graph)                    │
│  ├── Behavioral Signals (Velocity, Anomaly)                 │
│  └── Risk Fusion (0.55·Model + 0.30·Network + 0.15·Behavior)│
├─────────────────────────────────────────────────────────────┤
│  Evidence Layer                                             │
│  ├── Coordinated-Risk Detection (Cluster Analysis)          │
│  ├── Evidence Synthesis (Convergence Detection)             │
│  ├── Evidence Prioritization (PRIMARY/SUPPORTING/CONTEXTUAL)│
│  └── Investigation Path (Next-Best-Step Engine)             │
├─────────────────────────────────────────────────────────────┤
│  Investigation Copilot                                      │
│  ├── Evidence Context Builder (Bounded, Verified)           │
│  ├── Grounded System Prompt (Never Overrides Decisions)     │
│  ├── LLM Provider (OpenRouter / OpenAI / Null)              │
│  └── Deterministic Fallback (When LLM Unavailable)          │
└─────────────────────────────────────────────────────────────┘
```

### Deterministic vs. LLM Boundary

The deterministic RazorGuard systems remain the **source of truth** for:
- Risk scores
- Decisions (BLOCK/REVIEW/ALLOW)
- Evidence synthesis
- Investigation path
- Case status

The LLM provides **investigation assistance only** — summarizing, explaining, and answering questions. It never modifies case state, risk scores, or decisions.

---

## Features

| Feature | Description |
|---------|-------------|
| **Transaction Risk Scoring** | ML model + network + behavioral fusion |
| **Command Center** | Real-time operational dashboard |
| **Case Queue** | Prioritized investigation queue |
| **Case Investigation** | Full investigation workspace |
| **Investigator Assignment** | Assign cases to analysts |
| **Case Lifecycle** | OPEN → IN_REVIEW → ESCALATED → RESOLVED |
| **Audit Trail** | Append-only investigation history |
| **Network Intelligence** | Entity relationship analysis |
| **Risk Analytics** | Aggregate risk intelligence |
| **Evidence Graph V2** | Interactive entity exploration |
| **Coordinated-Risk Detection** | Multi-entity cluster analysis |
| **Evidence Prioritization** | PRIMARY/SUPPORTING/CONTEXTUAL tiers |
| **Investigation Path** | Next-best-investigation-step |
| **Investigation Timeline** | Visual audit history |
| **Investigation Copilot** | AI-assisted analysis |

---

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- npm

### Backend

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python -m razorguard.data.generate

# Train model
python -m razorguard.ml.train

# Evaluate model
python -m razorguard.ml.evaluate

# Start API server
uvicorn razorguard.api.app:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

### Environment Variables

```bash
# Optional: LLM Investigation Copilot
# OpenRouter (preferred — free tier available, no credit card):
OPENROUTER_API_KEY=sk-or-...             # OpenRouter API key (backend only)
OPENROUTER_MODEL=openrouter/free         # Free model router (default)
OPENROUTER_TIMEOUT=30                     # Request timeout in seconds

# OpenAI (alternative):
OPENAI_API_KEY=sk-...                    # OpenAI API key (backend only)
OPENAI_MODEL=gpt-4o-mini                 # Model to use
OPENAI_TIMEOUT=30                         # Request timeout in seconds
```

**Provider priority**: OpenRouter → OpenAI → Deterministic fallback.

If no API key is configured, the copilot uses deterministic evidence-grounded fallback answers.

**Security**: API keys are backend-only. They are never exposed to the React frontend.

### Tests

```bash
# Backend tests
pytest -q

# Frontend build check
cd frontend && npm run build
```

---

## API Overview

All endpoints are under `/api/v1`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/transactions/score` | POST | Score a transaction, auto-create case |
| `/network/summary` | GET | Graph-level network statistics |
| `/network/transaction/{id}` | GET | Transaction network intelligence |
| `/network/transaction/{id}/cluster` | GET | Coordinated-risk cluster |
| `/cases` | GET | List cases with filtering |
| `/cases/{id}` | GET | Get case details |
| `/cases/{id}/assign` | POST | Assign investigator |
| `/cases/{id}/transition` | POST | Transition case status |
| `/cases/{id}/audit` | GET | Get audit history |
| `/cases/{id}/intelligence` | GET | Full intelligence bundle |
| `/copilot/status` | GET | Copilot provider status |
| `/copilot/ask` | POST | Ask copilot a question |
| `/dashboard/summary` | GET | Dashboard metrics |
| `/dashboard/distribution` | GET | Risk distribution |
| `/dashboard/activity` | GET | Recent activity |
| `/dashboard/queue` | GET | Priority queue |
| `/analytics/overview` | GET | Risk analytics |

---

## Investigation Journey

```
Command Center
    ↓
High-risk Case (from queue)
    ↓
Investigation Page
    ├── Risk Summary Bar (score, decision, status)
    ├── Primary Reason & Evidence
    ├── Case Actions (status transitions)
    ├── Investigator Assignment
    ├── Evidence Graph V2 (interactive entity exploration)
    ├── Coordinated-Risk Evidence (prioritized tiers)
    ├── Investigation Path (next-best-step)
    ├── Investigation Timeline (audit history)
    └── Investigation Copilot (AI assistance)
    ↓
Investigator Action
    ↓
Audit Trail
```

---

## Demo Flow

### Recommended Demo Case

The strongest evidence is found in cases with:

1. **HIGH or CRITICAL risk** (risk score ≥ 70)
2. **BLOCK decision** (highest severity)
3. **Network intelligence** (shared device/merchant)
4. **Coordinated-risk cluster** (multiple connected accounts)
5. **Investigation path** (actionable next steps)

### Demo Steps

1. **Command Center**: Show operational overview with risk distribution
2. **Case Queue**: Navigate to a high-risk case
3. **Investigation Page**:
   - Show risk summary bar (score, decision, model probability)
   - Show primary reason and evidence
   - Show evidence graph with entity relationships
   - Show coordinated-risk evidence (PRIMARY/SUPPORTING/CONTEXTUAL)
   - Show investigation path (next-best-step)
   - Show timeline (audit history)
4. **Case Action**: Assign investigator, transition status
5. **Copilot**: Ask "Why was this case flagged?" or "What should I investigate next?"
6. **Network Intelligence**: Show entity relationships and cluster analysis
7. **Risk Analytics**: Show aggregate risk intelligence

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19, TypeScript, Vite, TanStack Query, Axios |
| Backend | Python 3.12, FastAPI, Pydantic, pandas |
| ML | scikit-learn (Logistic Regression) |
| Storage | Parquet (file-based) |
| LLM | OpenRouter / OpenAI (optional, via httpx) |

---

## Known Limitations

- **File-based storage**: Uses Parquet files (not a database). Suitable for demo/prototype, not production scale.
- **Single-server**: No distributed processing or horizontal scaling.
- **Synthetic data**: Demo data is generated, not real transaction data.
- **LLM dependency**: Copilot requires OpenRouter or OpenAI API key. Without either, deterministic fallback is used.
- **No authentication**: No user auth or role-based access control.
- **No real-time**: No WebSocket or streaming updates.

---

## License

Internal use only. Not for distribution.
