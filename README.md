# Local Face Search

CLI tool to find photos of a person in an event folder. Face detection and embedding run on-device; PostgreSQL stores only paths, bounding boxes, and embedding vectors—never image pixels.

| Doc | Role |
|-----|------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Goals, privacy rules, stack, data model, module contracts — read before changing code |
| [`PLAN.md`](PLAN.md) | Short build checklist and implementation steps |

## How it works

1. **Index** a folder of event photos (detect faces → ArcFace embeddings → upsert into Postgres).
2. **Search** with a reference photo of the person.
3. Get a ranked list of matching **image paths** on disk.

```
photo → MediaPipe BlazeFace (detect) → InsightFace ArcFace (embed) → PostgreSQL + pgvector (search)
```

Detection answers “where is a face?” Embedding + cosine search answer “who matches?”

## Privacy

- All ML runs locally on your machine.
- The database never receives JPEG/PNG bytes or face crop files—only paths, bbox integers, and float vectors.
- Prefer local Postgres so the whole pipeline stays offline.

See [ARCHITECTURE.md §2](ARCHITECTURE.md) for the full rules.

## Requirements

- Python 3.10+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) (Homebrew on macOS is the default path)
- BlazeFace model file: `blaze_face_full_range.tflite` in the project root (gitignored)

### Python packages

```bash
pip install mediapipe opencv-python numpy insightface onnxruntime "psycopg[binary]" pgvector python-dotenv
```

InsightFace downloads the `buffalo_l` model on first use (cached under your home directory).

## Postgres setup

1. Install PostgreSQL and enable pgvector (e.g. Homebrew `postgresql` + `pgvector`).
2. Create a database, e.g. `face_search`.
3. Apply the schema:

```bash
psql "$DATABASE_URL" -f schema.sql
```

4. Copy `.env.example` to `.env` and set your connection string:

```
DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/face_search
```

## Usage

### Index an event folder

```bash
python main.py index --folder /path/to/event-photos
python main.py index --folder /path/to/event-photos --reindex   # rewrite existing paths
```

Walks `.jpg`, `.jpeg`, `.png`, `.webp`. Without `--reindex`, already-indexed paths are skipped.

### Search by reference photo

```bash
python main.py search --image /path/to/person.jpg
python main.py search --image /path/to/person.jpg --threshold 0.35 --face-index 0
```

| Flag | Meaning |
|------|---------|
| `--threshold` | Minimum cosine similarity (default `0.35`; tune ~0.3–0.5 for your album) |
| `--face-index` | Which face in the reference image to use (default: largest) |
| `--limit` | Max face rows fetched from pgvector before filtering (default `50`) |

Results are distinct photos ranked by best face match above the threshold.

## Project layout

```
main.py            # CLI: index / search
face_detect.py     # BlazeFace → bbox + crop
face_embed.py      # ArcFace → L2-normalized 512-d vector
db.py              # psycopg + pgvector upsert / search
schema.sql         # faces table + vector extension
.env.example       # DATABASE_URL template
ARCHITECTURE.md    # full design
PLAN.md            # build checklist
```