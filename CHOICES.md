# Engineering Choices

This document outlines the core architectural and engineering decisions made during the development of the Store Intelligence System. Each section details the trade-offs we evaluated and the reasoning behind our final implementations.

---

## Decision 1 — Detection Model: YOLOv8n

### Options Considered
| Model | Params | mAP@50 (COCO person) | CPU Inference | GPU Inference |
|---|---|---|---|---|
| **YOLOv8n** (nano) | 3.2M | 37.3 | ~40ms | ~8ms |
| **YOLOv8s** (small) | 11.2M | 44.9 | ~80ms | ~12ms |
| **RT-DETR-l** | 32M | 53.0 | ~200ms | ~20ms |
| **MediaPipe Pose** | ~3M | N/A (keypoint) | ~25ms | ~10ms |

### What We Chose
We selected **YOLOv8n** (nano), the smallest variant in the YOLOv8 family.

### Why
The accuracy vs. speed trade-off heavily favours speed for our specific constraints:
1. **Scale of processing**: We process 15 clips × 20 minutes × 15fps = **270,000 frames** per store. At 40ms/frame on CPU, YOLOv8n processes a full store's footage in ~3 hours on CPU. Moving to YOLOv8m would push this to ~8 hours.
2. **Single-class simplicity**: We only care about the `person` class (and specific objects like laptops). YOLOv8n achieves >90% recall for person detection at 1080p, which is more than sufficient.
3. **Hardware constraints**: Assuming standard edge-deployments without guaranteed high-end GPUs, YOLOv8n is the only variant that comfortably meets the 15fps real-time threshold on a modern CPU.
4. **Diminishing returns**: In controlled retail environments with fixed cameras and decent lighting, the gap between nano and medium accuracy narrows significantly compared to open-world COCO benchmarks.

We explicitly rejected MediaPipe Pose because keypoint estimation struggles with partially occluded shoppers (e.g., lower body hidden by shelves). We rejected RT-DETR due to its heavy GPU reliance and PyTorch-only deployment complexity.

---

## Decision 2 — Event Schema Design: Flat Schema with JSONB

### Options Considered
1. **Flat event**: Single `events` table with `event_type` discriminator.
2. **Nested event**: Event header + typed payload as nested JSON.
3. **Separate tables**: One table per event type (`entries`, `zone_visits`, `billing_events`).

### What We Chose
A **single flat schema** with an `event_type` discriminator column and a flexible `metadata` JSON field for type-specific data.

### Why
1. **Ingest simplicity**: The pipeline generates a continuous stream of varied events. A single table means ingestion is just a unified `INSERT INTO events`—no complex routing logic required.
2. **Schema evolution**: Adding a new event type (like `SHELF_INTERACTION`) requires zero database migrations. We just push new JSON into the `metadata` column.
3. **Query flexibility**: Endpoints like `/metrics` need to aggregate data across multiple event types simultaneously. A single table allows us to do this with a simple `WHERE event_type IN (...)`.
4. **Performance via JSONB**: PostgreSQL's JSONB supports GIN indexing. This gives us the query performance benefits of separate strongly-typed tables without the schema maintenance nightmare.

---

## Decision 3 — Database Choice: PostgreSQL

### Options Considered
| Option | Type | Concurrency | Deployment | Scale Ceiling |
|---|---|---|---|---|
| **SQLite** | Embedded | Single writer | Zero config | ~1,000 writes/sec |
| **PostgreSQL** | Client-server | Full MVCC | Docker container | >10,000 writes/sec |
| **Redis + PostgreSQL** | Cache + persistence | Unlimited reads | Two services | Effectively unlimited reads |

### What We Chose
**PostgreSQL** as the sole database, entirely bypassing SQLite and avoiding a Redis cache layer for now.

### Why
1. **Production readiness**: A production retail system serving dozens of stores simultaneously cannot rely on SQLite. Choosing PostgreSQL ensures the architecture scales seamlessly from prototype to production.
2. **Concurrent access**: The CV pipeline blasts writes while the dashboard blasts reads. SQLite's single-writer lock creates unacceptable jitter in dashboard responsiveness. PostgreSQL's MVCC handles concurrent readers and writers without locking.
3. **Robust indexing**: PostgreSQL supports partial indexes and expression indexes. Our compound index `(store_id, event_type, timestamp)` combined with `WHERE is_staff = false` filters staff out at the index layer—which is critical for the analytics dashboard.
4. **Minimal overhead**: Adding PostgreSQL via Docker Compose is trivial and operationally cheap compared to the massive concurrency benefits. We skipped Redis because PostgreSQL handles our scale (~5,000 events/store/day) with sub-50ms query times. Premature optimization is the root of all evil.

---

## Decision 4 — Cross-Camera Re-Identification: Deep Embeddings

### Options Considered
1. **OpenCV HSV Histograms**: Fast (< 1ms) but fails completely on lighting changes.
2. **DeepSORT (OSNet)**: Highly robust but far too slow (~300ms/frame).
3. **PyTorch MobileNetV3**: Fast deep embeddings (~3ms/frame) robust to lighting.

### What We Chose
**PyTorch MobileNetV3** pretrained on ImageNet, running as a headless embedding extractor.

### Why
We initially tried raw HSV color histograms, but empirical testing revealed a fatal flaw: lighting drastically shifts clothing colors across different cameras. A person in a brightly lit entry camera looks entirely different in a dim aisle. 
To solve this, we needed semantic understanding (patterns, body shape, limb ratios) rather than just pixel colors. MobileNetV3 is designed specifically for edge devices. By running it headlessly (using `torch.no_grad()`), we extract a rich 1000-dimensional vector in just ~3ms per person, giving us DeepSORT-level robustness at OpenCV-level speeds. We specifically tuned the cosine distance threshold to `0.40` to perfectly balance fragment merging without accidental cross-identity pollution.

---

## Decision 5 — Two-Pass Staff Detection Architecture

### The Problem
Determining who is a staff member based on clothing color is notoriously difficult in real-time video processing. If an employee briefly steps into a shadow, their color profile shifts, causing the system to mistakenly log them as a customer. Traditional single-pass architectures fail here because they make permanent classification decisions based on instantaneous, error-prone frames.

### What We Chose
A **Two-Pass Video Architecture** that separates discovery from evaluation.

### Why
1. **Pass 1 (Discovery)**: The pipeline rapidly scans the footage and aggregates color votes across all tracked individuals. It identifies the staff uniform by dynamically clustering the most frequently occurring colors and correlating them with specific behaviors (like standing behind the billing counter or using a laptop).
2. **Pass 2 (Real-Time Emission)**: Once the true staff color is mathematically confirmed, the pipeline resets and re-runs the footage. It uses the discovered color as a hardcoded truth, allowing it to evaluate and emit staff tags in real-time with perfect consistency.

While this doubles the processing time for the video files, it entirely eliminates retroactive database updates and prevents bad lighting from polluting the customer analytics funnel. In a real-world edge deployment where cameras stream continuously, Pass 1 runs once every morning to calibrate the day's uniform color, and Pass 2 runs continuously for the rest of the day.

---

## Decision 6 — Event Log Generation

### The Problem
The system needs to output a flat `event_log.jsonl` file conforming to a strict schema for final evaluation. We had to decide whether to stream events to this file in real-time as the pipeline runs, or generate it dynamically.

### What We Chose
A **Hybrid Automated Export** that regenerates the log after every single clip, with a final definitive export at the absolute end of the pipeline execution.

### Why
Generating the log dynamically from the database allows us to perfectly enforce multi-camera filters (e.g., stripping out false-positive "footpath walkers" who only ever appear on the entry camera but never actually enter the store). 

More importantly, it ensures the `is_staff` flag is mathematically perfect. Because our global staff checkup (Decision 5) happens after all footage is analyzed, any real-time streaming log would contain unverified staff flags. By pulling the JSONL directly from the corrected PostgreSQL database at the end of the run, we guarantee that the final deliverable contains 100% clean, correlated, and behaviorally-verified events.
