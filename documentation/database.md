# Database Layer

## Overview

DRAW uses a SQL database (via SQLAlchemy) as both a task queue and an audit log for the continuous prediction pipeline. The database tracks every DICOM study from ingestion through prediction and delivery.

## Schema

```mermaid
erDiagram
    DICOMLOG {
        bigint id PK "Auto-increment"
        varchar(256) series_name UK "DICOM SeriesInstanceUID"
        varchar(1024) input_path "Absolute path to input DICOM directory"
        varchar(1024) output_path "Path to output RT-Struct (set after prediction)"
        enum status "INIT | STARTED | PREDICTED | SENT"
        enum model "Cancer-site model name (e.g., TSPrime)"
        datetime created_on "Insertion timestamp"
    }
```

## Status Lifecycle

```mermaid
sequenceDiagram
    participant W as Watcher Process
    participant DB as Database
    participant P as Predictor Process

    W->>DB: INSERT (status=INIT)
    P->>DB: SELECT WHERE status=INIT (dequeue)
    P->>DB: UPDATE status=STARTED
    Note over P: Run nnU-Net inference
    P->>DB: UPDATE status=PREDICTED, output_path=...
    P->>DB: UPDATE status=SENT
```

## Supported Databases

The `DB_URL` in `env.draw.yml` accepts any SQLAlchemy-compatible connection string:

| Database | Connection String | Notes |
|----------|------------------|-------|
| SQLite | `sqlite:///draw.db` | Good for single-machine deployments |
| PostgreSQL | `postgresql://user:pass@host:5432/draw` | Recommended for production |
| MySQL | `mysql+pymysql://user:pass@host:3306/draw` | Requires `pymysql` package |

## Connection Pool Settings

Configured in `draw/dao/common.py`:

```python
DB_ENGINE = create_engine(
    DB_CONFIG["URL"],
    echo=False,
    isolation_level="READ UNCOMMITTED",
    pool_size=10,
    max_overflow=20,
    pool_timeout=100,
)
```

- **Isolation level**: `READ UNCOMMITTED` — acceptable because the two processes operate on non-overlapping record states (watcher writes `INIT`, consumer reads `INIT` and updates to `STARTED/PREDICTED/SENT`)
- **Pool size**: 10 connections (handles the two-process pipeline with headroom for monitoring queries)

## Queue Operations

The `DBConnection` class provides queue semantics:

| Method | Description |
|--------|-------------|
| `enqueue(records)` | Insert new DicomLog records (status=INIT) |
| `dequeue(model)` | Fetch and mark top INIT records as STARTED |
| `top(model, status)` | Peek at records by model name and status |
| `exists(series_name)` | Check if a SeriesInstanceUID already exists (deduplication) |
| `update_status_by_id(dcm, status)` | Transition a record to a new status |
| `update_record_by_series_name(...)` | Set output_path and status after prediction |

## Migrations

Database schema migrations are managed via Alembic. Migration scripts live in `draw/alembic/versions/`.

To run migrations:
```bash
alembic upgrade head
```

## Monitoring Queries

### Check queue depth
```sql
SELECT model, status, COUNT(*) 
FROM dicomlog 
GROUP BY model, status;
```

### Find stuck predictions
```sql
SELECT * FROM dicomlog 
WHERE status = 'STARTED' 
AND created_on < NOW() - INTERVAL '1 hour';
```

### Daily throughput
```sql
SELECT DATE(created_on), COUNT(*) 
FROM dicomlog 
WHERE status = 'SENT' 
GROUP BY DATE(created_on) 
ORDER BY 1 DESC;
```
