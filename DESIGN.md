# System Design Document

## System Architecture

The Store Intelligence System is a multi-layered pipeline that transforms raw CCTV footage into actionable retail analytics. The architecture follows a clean separation of concerns: a detection layer processes video into structured events, an API layer persists and queries those events, and a dashboard layer visualises the results in real time.

The system is designed to process footage from up to 40 retail stores, each producing approximately 15 clips of 20 minutes each. Events are generated at the clip level and ingested in batches into a central database. All analytics — visitor counts, conversion funnels, heatmaps, and anomaly detection — are computed on-demand from the stored event stream rather than pre-aggregated, ensuring consistency and flexibility.

### Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                        STORE INTELLIGENCE SYSTEM                      │
│                                                                       │
│  ┌─────────────┐    ┌─────────────────────┐    ┌──────────────────┐  │
│  │             │    │   DETECTION LAYER    │    │                  │  │
│  │  CCTV Clips │───▶│                     │───▶│  Event Stream    │  │
│  │  (raw .mp4) │    │  YOLOv8n detector   │    │  (JSON batches)  │  │
│  │             │    │  MobileNetV3 Re-ID  │    │                  │  │
│  └─────────────┘    │  Zone polygon mapper │    └────────┬─────────┘  │
│                     └─────────────────────┘             │            │
│                                                          │            │
│                     ┌─────────────────────┐             │            │
│                     │    API LAYER         │◀────────────┘            │
│                     │                     │                          │
│                     │  POST /events/ingest│                          │
│                     │  GET  /stores/*/    │                          │
│                     │    metrics          │                          │
│                     │    heatmap          │                          │
│                     │    funnel           │                          │
│                     │    anomalies        │                          │
│                     │  GET  /health       │                          │
│                     └──────────┬──────────┘                          │
│                                │                                     │
│                     ┌──────────▼──────────┐    ┌──────────────────┐  │
│                     │   STORAGE LAYER     │    │  DASHBOARD LAYER │  │
│                     │                     │    │                  │  │
│                     │  PostgreSQL 15      │◀───│  Streamlit app   │  │
│                     │  • events table     │    │  (polls API)     │  │
│                     │  • pos_transactions │    │                  │  │
│                     │  • indexed by       │    │  Live metrics    │  │
│                     │    store+timestamp  │    │  Zone heatmap    │  │
│                     └─────────────────────┘    │  Anomaly alerts  │  │
│                                                │  Funnel chart    │  │
│                                                └──────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Detection Layer

### Model Choice: YOLOv8n

We use YOLOv8n (nano) from Ultralytics for person detection. The nano variant was chosen because the task is limited to single-class person detection at 1080p resolution — a problem that even the smallest YOLO model handles reliably with >90% mAP. The inference speed (~8ms per frame on GPU, ~40ms on CPU) comfortably meets the 15fps processing requirement.

We evaluated YOLOv8s (small) and RT-DETR as alternatives. YOLOv8s provides marginally better accuracy (+1.5 mAP) but doubles inference time, which becomes significant across 15 clips × 20 minutes × 15fps = 270,000 frames. RT-DETR's transformer architecture offers excellent accuracy but requires GPU acceleration and adds deployment complexity.

### Tracking Approach: IoU Tracker with Deep Learning Re-ID

We implemented a custom IoU-based tracker augmented with a PyTorch-based Deep Learning Re-Identification (Re-ID) model. Pure IoU tracking (matching detections between frames by bounding-box overlap) is fast and effective when the frame rate is high enough that people don't move far between frames.

To address cross-camera tracking and re-identification after brief occlusions, we integrated a `MobileNetV3` embedding model. When a person is detected, the model extracts a 1000-dimensional semantic feature vector (embedding) of their appearance. We compare these embeddings against a global feature buffer using cosine similarity. If the cosine distance falls below our configured threshold (`0.40`), the system successfully merges their IDs across different cameras, ensuring a single unified ID throughout the entire store.

We initially considered a simple OpenCV HSV color histogram approach to save computational cost. However, empirical testing showed that color histograms were too brittle when dealing with the drastically different lighting conditions and camera angles present across the CCTV feeds. The `MobileNetV3` approach provides incredible semantic accuracy (flawlessly unifying customers and employees across up to 4 different cameras) while remaining lightweight enough to run efficiently on a CPU.

### Two-Pass Staff Detection

A major engineering hurdle was distinguishing staff from customers using uniform colors. A single-pass system that evaluates a person's color at the moment they enter the frame is fundamentally flawed; a staff member stepping into a shadow will be permanently misclassified as a customer, destroying analytics accuracy.

To solve this, we designed a **Two-Pass Architecture**:
1. **Discovery Pass**: We rapidly scan the footage (up to 3 minutes), aggregate color votes for every tracked ID, and correlate them with behavioral heuristics (e.g., spending time behind the billing counter or interacting with a laptop) to definitively identify the exact RGB/HSV signature of the employee uniform for that specific store and lighting environment.
2. **Real-Time Pass**: The pipeline resets and runs in real-time, using the discovered uniform color as a strict, hardcoded template. As people move through the store, they are continuously evaluated against this hardcoded template, preventing momentary shadows from polluting the dataset.

### Zone Detection

Zones are defined as polygons in a per-store configuration file. Each frame, we compute the centroid (foot-point) of each detected person's bounding box and test whether it falls inside any zone polygon using the `shapely` library's `Point.within(Polygon)` check. Zone transitions (entering/exiting a zone) generate `ZONE_VISIT` and `ZONE_EXIT` events.

This approach is more robust than grid-based zone assignment because real store layouts have irregular shapes.

---

## API Layer

### Why FastAPI

FastAPI was chosen for several reasons:
1. **Automatic validation**: Pydantic models validate every request body and return structured 422 errors for malformed data — critical for the ingest endpoint where partial success handling is required.
2. **Async support**: While we use synchronous SQLAlchemy for simplicity, FastAPI's async foundation means we can switch to async DB drivers (asyncpg) without rewriting route handlers.
3. **OpenAPI/Swagger**: Auto-generated API documentation at `/docs` makes integration testing and manual exploration straightforward.
4. **Dependency injection**: The `Depends(get_db)` pattern makes database sessions trivially swappable in tests (in-memory SQLite) vs. production (PostgreSQL).

### Endpoint Design

All analytics endpoints follow a consistent pattern:
```
GET /stores/{store_id}/{resource}
```
This REST-style design makes the API intuitive and cacheable. The `store_id` path parameter scopes every query, preventing cross-store data leakage and enabling per-store caching in future.

### Session-Based Computation

Metrics, funnels, and anomalies are computed **on-demand** from raw events rather than maintained as materialised aggregates. This design was a deliberate trade-off:
- **Pro**: No data staleness, no sync bugs, simpler codebase
- **Con**: Slower queries for stores with very high event volumes

For the target scale (40 stores, ~5,000 events per store per day), on-demand computation with proper database indexing (compound index on `store_id` + `timestamp`) responds within 50ms.

---

## Data Flow

```
  Video Frame                          Database                     Client
  ──────────                          ────────                     ──────
      │                                                              
      ▼                                                              
  [YOLOv8n Detect]                                                   
      │ bbox list                                                    
      ▼                                                              
  [IoU Tracker + Re-ID]                                                      
      │ track_id assigned                                            
      ▼                                                              
  [Zone Mapper]                                                      
      │ zone determined                                              
      ▼                                                              
  [Event Generator]                                                  
      │ EventIn dict                                                 
      ▼                                                              
  [Batch Buffer]                                                     
      │ 50-event batches                                             
      ▼                                                              
  POST /events/ingest  ──────▶  INSERT INTO events ──────────────────
                                      │                              
                                      ▼                              
                               GET /stores/X/metrics  ◀──────── Dashboard
                               GET /stores/X/heatmap  ◀──────── Dashboard
                               GET /stores/X/funnel   ◀──────── Dashboard
                               GET /stores/X/anomalies◀──────── Dashboard
                                      │
                                      ▼
                             [Pipeline Complete]
                                      │
                                      ▼
                        [JSONL Export (event_log.jsonl)]

```

1. **Detection**: Each video frame is processed by YOLOv8n to produce person bounding boxes.
2. **Tracking**: Bounding boxes are matched to persistent tracks using IoU overlap and MobileNetV3 embeddings.
3. **Zone mapping**: Track centroids are tested against store zone polygons.
4. **Event generation**: State transitions (entry, exit, zone change, queue join) produce structured events.
5. **Batch ingestion**: Events are buffered and POSTed in batches of up to 50 to the API.
6. **Storage**: The API validates events, deduplicates by `event_id`, and inserts into PostgreSQL.
7. **Query**: Dashboard and external clients query analytics endpoints which compute results from stored events on demand.
8. **Automated Export**: Upon pipeline completion and final staff re-identification, the clean records are automatically mapped and exported to the deliverable `event_log.jsonl` schema.

---

## AI-Assisted Decisions

We used LLMs extensively throughout the project — primarily Claude and Gemini — as a sounding board for architectural decisions. Here are three cases where their suggestions shaped (or didn't shape) the final system.

### 1. Tracker Selection

Early on, we asked for guidance on person tracking approaches. The suggestion was DeepSORT, which made sense given its robust re-identification capabilities. We started simpler though — a basic IoU tracker with HSV color histogram matching. That worked fine within a single camera, but completely fell apart across different cameras. The lighting difference between, say, a bright entrance cam and a dim aisle cam was enough to make the same person's color histogram unrecognizable. We eventually landed on MobileNetV3 as a feature extractor — it gave us DeepSORT-level semantic matching at a fraction of the computational cost. The LLM's initial instinct about needing deep features was right, but we found a much lighter way to get there.

### 2. Staff Classification

For identifying employees, the initial suggestion was CLIP zero-shot classification — prompts like "a person wearing a store uniform" vs "a customer." We tried it. It added ~200ms per crop and gave inconsistent results when lighting shifted across the store's cameras. What actually worked was our two-pass HSV color voting system: scan the first 3 minutes of footage to dynamically discover the uniform color, then use that as a strict template on the second pass. This runs in under 1ms per person and is far more reliable in practice than a vision-language model for this specific task.

### 3. Database Selection

The recommendation here was SQLite for simplicity — no extra Docker service, single-file database, trivial backups. We disagreed. Our pipeline writes events continuously while the dashboard reads metrics simultaneously. SQLite's single-writer lock would create noticeable lag in dashboard responsiveness during active processing. PostgreSQL's MVCC handles concurrent reads and writes without contention, and the Docker overhead is genuinely minimal (one extra service in docker-compose, ~100MB image). For a system that needs to scale to 40 stores, PostgreSQL was the obvious choice.

---

## Performance Considerations

| Component | Throughput | Latency |
|---|---|---|
| YOLOv8n (CPU) | 25 fps | ~40ms/frame |
| YOLOv8n (GPU) | 120+ fps | ~8ms/frame |
| IoU Tracker + Re-ID | ~300 fps | ~3ms/frame |
| Event Ingest (500 batch) | ~2,000 events/sec | <250ms |
| Metrics Query | — | <50ms |
| Heatmap Query | — | <30ms |
| Anomaly Query | — | <80ms |

The bottleneck is always the detection model. All downstream components (tracking, zone mapping, API ingest, queries) operate at least an order of magnitude faster than detection.

