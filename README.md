# 🏪 Store Intelligence System

**AI-Powered Retail Analytics from Raw CCTV Footage**

An end-to-end pipeline that transforms raw CCTV footage into actionable store analytics: real-time visitor tracking, zone heatmaps, conversion funnels, and anomaly detection — all exposed through production-grade REST APIs and a live dashboard.

> Built for the Purplle Tech Challenge 2026 — Round 2

---

## 📋 Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Docker & Docker Compose | 24.0+ / v2 |
| Git | 2.40+ |
| NVIDIA GPU (optional) | CUDA 12.x drivers — speeds up YOLOv8 inference ~10× |

---

## 🚀 Quick Start

Get the system running in under 5 commands:

```bash
# 1. Clone and enter the project
git clone https://github.com/amalkuruvath-del/store-intelligence.git 

cd store-intelligence

# 2. Build and start all services (API + PostgreSQL + Dashboard)
docker compose up --build -d

# 3. Wait for services to be healthy
docker compose ps   # All services should show "healthy"

# 4. Prepare the dataset
# Keep the store folder (containing the .mp4 files and store_layout.json) 
# directly into the existing 'data/' directory in this repo.

# 5. Install pipeline dependencies
python -m venv venv
source venv/bin/activate  # (Or `.\venv\Scripts\activate` on Windows)
pip install -r pipeline/requirements.txt

# 6. Run the detection pipeline against the CCTV clips
# Usage:run.ps1 -DataDir "../data/ [PATH_TO_YOUR_DATA_FOLDER] [STORE_ID]
cd pipeline 
bash run.sh ../data/Store_1 STORE_01
.\run.ps1 -DataDir "../data/Store 1" -StoreId "STORE_01"

# 7. Verify — query store metrics
curl http://localhost:8000/stores/STORE_BLR_002/metrics | python -m json.tool
```

- **API**: `http://localhost:8000`
- **Dashboard**: `http://localhost:8501`
- **API Docs (Swagger)**: `http://localhost:8000/docs`

---

## 📡 API Documentation

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "db_connected": true,
  "last_event_per_store": {"STORE_BLR_002": "2026-03-03T14:38:12Z"},
  "stale_feeds": [],
  "uptime_seconds": 3600.0
}
```

---

### Ingest Events

```bash
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '[
    {
      "event_id": "550e8400-e29b-41d4-a716-446655440000",
      "store_id": "STORE_BLR_002",
      "camera_id": "CAM_ENTRY_01",
      "visitor_id": "VIS_c8a2f1",
      "event_type": "ENTRY",
      "timestamp": "2026-03-03T14:22:10Z",
      "zone_id": null,
      "dwell_ms": 0,
      "is_staff": false,
      "confidence": 0.91,
      "metadata": {
        "queue_depth": null,
        "sku_zone": null,
        "session_seq": 1
      }
    }
  ]'
```

```json
{"accepted": 1, "rejected": 0, "errors": []}
```

- Accepts batches of up to **500 events**
- **Idempotent** by `event_id` — safe to retry
- **Partial success** — valid events are ingested, malformed ones returned in `errors`

---

### Store Metrics

```bash
curl http://localhost:8000/stores/STORE_BLR_002/metrics
```

```json
{
  "store_id": "STORE_BLR_002",
  "date": "2026-03-03",
  "unique_visitors": 142,
  "conversion_rate": 0.32,
  "avg_dwell_per_zone": {"SKINCARE": 45000, "COSMETICS": 32000},
  "current_queue_depth": 3,
  "abandonment_rate": 0.12
}
```

---

### Zone Heatmap

```bash
curl http://localhost:8000/stores/STORE_BLR_002/heatmap
```

Returns zone visit frequency + avg dwell, **normalised 0–100**. Includes `data_confidence: "low"` if fewer than 20 sessions.

---

### Conversion Funnel

```bash
curl http://localhost:8000/stores/STORE_BLR_002/funnel
```

```json
{
  "store_id": "STORE_BLR_002",
  "stages": [
    {"stage": "Entry", "count": 142, "drop_off_pct": 0.0},
    {"stage": "Zone Visit", "count": 128, "drop_off_pct": 9.86},
    {"stage": "Billing Queue", "count": 67, "drop_off_pct": 47.66},
    {"stage": "Purchase", "count": 45, "drop_off_pct": 32.84}
  ]
}
```

---

### Anomaly Detection

```bash
curl http://localhost:8000/stores/STORE_BLR_002/anomalies
```

Detects: `BILLING_QUEUE_SPIKE`, `CONVERSION_DROP`, `DEAD_ZONE`. Each includes `severity` (INFO/WARN/CRITICAL) and `suggested_action`.

---

## 🧪 Running Tests

```bash
# Run full test suite with coverage
docker compose exec api pytest tests/ --cov=app -v

# Run locally (requires virtualenv with dependencies)
pytest tests/ --cov=app --cov-report=term-missing -v

# Run a specific test file
pytest tests/test_api.py -v
```

---

## 📊 Dashboard

Available at `http://localhost:8501` when running via Docker Compose.

Features:
- **Store Selector** — dropdown to switch between stores
- **Metric Cards** — unique visitors, conversion rate, avg dwell, queue depth, abandonment
- **Zone Heatmap** — colour-coded grid showing zone activity intensity
- **Active Anomalies** — real-time alerts with severity badges
- **Conversion Funnel** — horizontal bar chart (Entry → Zone → Billing → Purchase)
- **Auto-Refresh** — polls the API every 5 seconds

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  CCTV Clips │────▶│ Detection Pipeline│────▶│ Event Stream │
│  (15 clips) │     │  YOLOv8n + PyTorch│     │  (JSON batch)│
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                      │
                                                      ▼
                    ┌──────────────────┐     ┌──────────────┐
                    │   PostgreSQL DB  │◀────│  FastAPI App  │
                    │  (events, POS)   │     │  REST API     │
                    └──────────────────┘     └──────┬───────┘
                                                      │
                                                      ▼
                                             ┌──────────────┐
                                             │  Streamlit   │
                                             │  Dashboard   │
                                             └──────────────┘
```

---

## 📁 Project Structure

```
store-intelligence/
├── app/                        # FastAPI application
│   ├── main.py                 # App entrypoint, lifespan, logging middleware
│   ├── database.py             # SQLAlchemy engine, session, Base
│   ├── models.py               # Pydantic schemas + SQLAlchemy ORM models
│   ├── ingestion.py            # POST /events/ingest
│   ├── metrics.py              # GET /stores/{id}/metrics + heatmap
│   ├── funnel.py               # GET /stores/{id}/funnel
│   ├── anomalies.py            # GET /stores/{id}/anomalies
│   └── health.py               # GET /health
├── pipeline/                   # CV detection pipeline
│   ├── config.py               # Centralised config + store_layout.json parser
│   ├── detect.py               # Main YOLOv8 detection loop (CLI)
│   ├── tracker.py              # IoU tracker + MobileNetV3 Re-ID + staff classification
│   ├── emit.py                 # Event construction + batch POST
│   └── run.sh                  # One-command pipeline runner
├── dashboard/
│   └── app.py                  # Streamlit live dashboard
├── tests/
│   ├── conftest.py             # Shared fixtures (SQLite in-memory, TestClient)
│   ├── test_api.py             # Ingestion, idempotency, degradation tests
│   ├── test_metrics.py         # Metrics, heatmap, funnel tests
│   └── test_anomalies.py       # Anomaly detection tests
├── DESIGN.md                   # Architecture + AI-assisted decisions
├── CHOICES.md                  # 5 engineering decisions with full reasoning
├── docker-compose.yml          # Services: api, db, dashboard
├── Dockerfile                  # python:3.11-slim single-stage build
├── requirements.txt            # Pinned Python dependencies
└── README.md                   # This file
```

---

## 📄 License

This project was built for the Purplle Tech Challenge 2026 — Round 2.
