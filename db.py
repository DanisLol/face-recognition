import os
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()  # reads .env into os.environ


#reading url and connecting to enable vectors
def get_connection():
    """Open a Postgres connection with pgvector types registered."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (check .env)")

    conn = psycopg.connect(url)
    register_vector(conn)  # required so vector columns work
    return conn


def upsert_faces(rows: list[dict]) -> int:
    """
    Insert or update face rows. Each dict needs:
    image_path, image_name, face_index, bbox_x/y/w/h, embedding (512 floats).
    Returns number of rows written.
    """
    sql = """
        INSERT INTO faces (
            image_path, image_name, face_index,
            bbox_x, bbox_y, bbox_w, bbox_h, embedding
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (image_path, face_index) DO UPDATE SET
            image_name = EXCLUDED.image_name,
            bbox_x = EXCLUDED.bbox_x,
            bbox_y = EXCLUDED.bbox_y,
            bbox_w = EXCLUDED.bbox_w,
            bbox_h = EXCLUDED.bbox_h,
            embedding = EXCLUDED.embedding
    """

    if rows == []:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            for r in rows:
                emb = r["embedding"]
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
                cur.execute(sql, (
                    r["image_path"],
                    r["image_name"],
                    r["face_index"],
                    r["bbox_x"],
                    r["bbox_y"],
                    r["bbox_w"],
                    r["bbox_h"],
                    emb,
                ))
        conn.commit()

    return len(rows)


def search(query_embedding, limit: int = 50, threshold: float | None = None) -> list[dict]:
    """Return faces ranked by cosine similarity to query_embedding."""
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()

    sql = """
        SELECT image_path, image_name, face_index,
               1 - (embedding <=> %s::vector) AS similarity
        FROM faces
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_embedding, query_embedding, limit))
            rows = cur.fetchall()

    results = []
    for image_path, image_name, face_index, similarity in rows:
        if threshold is not None and similarity < threshold:
            continue
        results.append({
            "image_path": image_path,
            "image_name": image_name,
            "face_index": face_index,
            "similarity": float(similarity),
        })
    return results
