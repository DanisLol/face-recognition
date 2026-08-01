"""
CLI entry for local face search.

Orchestrates face_detect → face_embed → db. Does not store image pixels
in Postgres—only paths, bbox ints, and embedding vectors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from db import get_connection, search as db_search, upsert_faces
from face_detect import detect_faces
from face_embed import embed_face

# Extensions we treat as photos when walking an event folder.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ArcFace cosine similarity default; tune empirically on your album (~0.3–0.5).
DEFAULT_THRESHOLD = 0.35


def faces_to_rows(image_path: str) -> list[dict]:
    """
    Detect faces in one image, embed each crop, and build DB row dicts.

    Returns an empty list when no faces are found. Never includes image bytes—
    only path metadata, bbox, and the L2-normalized embedding vector.
    """
    faces = detect_faces(image_path)
    resolved = str(Path(image_path).resolve())
    name = Path(image_path).name
    rows: list[dict] = []

    for f in faces:
        # ArcFace embedding; already L2-normalized inside face_embed.embed_face
        emb = embed_face(f.crop)
        rows.append(
            {
                "image_path": resolved,
                "image_name": name,
                "face_index": f.face_index,
                "bbox_x": f.bbox_x,
                "bbox_y": f.bbox_y,
                "bbox_w": f.bbox_w,
                "bbox_h": f.bbox_h,
                "embedding": emb,
            }
        )
    return rows


def collect_images(folder: Path) -> list[Path]:
    """
    Recursively collect image files under folder with known photo extensions.
    """
    images: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def path_already_indexed(image_path: str) -> bool:
    """
    Return True if at least one face row already exists for this image_path.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM faces WHERE image_path = %s LIMIT 1",
                (image_path,),
            )
            return cur.fetchone() is not None


def delete_faces_for_path(image_path: str) -> None:
    """
    Remove all face rows for image_path so --reindex can rewrite cleanly.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM faces WHERE image_path = %s", (image_path,))
        conn.commit()


def cmd_index(folder: Path, reindex: bool) -> int:
    """
    Walk a photo folder: detect → embed → upsert into Postgres.

    Without --reindex, skip images that already have rows. With --reindex,
    delete existing rows for each path then rewrite.
    """
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1

    images = collect_images(folder)
    if not images:
        print(f"No images found under {folder}")
        return 0

    files_ok = 0
    faces_total = 0
    skipped = 0
    errors = 0

    for img_path in images:
        resolved = str(img_path.resolve())
        print(f"Processing {img_path} ...")

        # Skip already-indexed paths unless the user asked to reindex
        if not reindex and path_already_indexed(resolved):
            print(f"  skip (already indexed)")
            skipped += 1
            continue

        try:
            if reindex:
                delete_faces_for_path(resolved)

            rows = faces_to_rows(resolved)
            if not rows:
                print(f"  no faces found")
                files_ok += 1
                continue

            n = upsert_faces(rows)
            print(f"  indexed {n} face(s)")
            files_ok += 1
            faces_total += n
        except Exception as exc:
            # Keep going on bad/unreadable files (ARCHITECTURE error table)
            print(f"  error: {exc}", file=sys.stderr)
            errors += 1

    print(
        f"Done. files_ok={files_ok} faces={faces_total} "
        f"skipped={skipped} errors={errors}"
    )
    return 0 if errors == 0 else 1


def pick_query_face(faces: list, face_index: int | None):
    """
    Choose which detected face to use for search.

    Default: largest bbox by area. Or an explicit --face-index.
    """
    if face_index is not None:
        for f in faces:
            if f.face_index == face_index:
                return f
        raise ValueError(
            f"face-index {face_index} not found "
            f"(valid: {[f.face_index for f in faces]})"
        )
    # Largest box is usually the primary subject in a reference selfie
    return max(faces, key=lambda f: f.bbox_w * f.bbox_h)


def distinct_best_per_photo(results: list[dict]) -> list[dict]:
    """
    Collapse face-level hits to one row per image_path (best similarity wins).
    """
    best: dict[str, dict] = {}
    for row in results:
        path = row["image_path"]
        if path not in best or row["similarity"] > best[path]["similarity"]:
            best[path] = row
    # Rank photos by that best score, highest first
    return sorted(best.values(), key=lambda r: r["similarity"], reverse=True)


def cmd_search(
    image: Path,
    threshold: float,
    face_index: int | None,
    limit: int,
) -> int:
    """
    Embed a reference face and print ranked matching photo paths from Postgres.
    """
    if not image.is_file():
        print(f"Not a file: {image}", file=sys.stderr)
        return 1

    try:
        faces = detect_faces(str(image))
    except Exception as exc:
        print(f"Detection failed: {exc}", file=sys.stderr)
        return 1

    if not faces:
        print("No faces found in reference image.", file=sys.stderr)
        return 1

    try:
        chosen = pick_query_face(faces, face_index)
        # Use a higher fetch limit so post-filter / distinct collapse still has candidates
        emb = embed_face(chosen.crop)
        raw = db_search(emb, limit=limit, threshold=threshold)
    except Exception as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    ranked = distinct_best_per_photo(raw)
    if not ranked:
        print("No matches above threshold.")
        return 0

    print(f"Matches (threshold={threshold}, query face_index={chosen.face_index}):")
    for i, row in enumerate(ranked, start=1):
        print(
            f"{i:3d}. {row['similarity']:.4f}  "
            f"{row['image_path']}  (face {row['face_index']})"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Build argparse with index and search subcommands.
    """
    parser = argparse.ArgumentParser(
        description="Local face search: index a photo folder, search by reference face.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- index ---
    p_index = sub.add_parser("index", help="Detect/embed faces in a folder into Postgres")
    p_index.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Path to event photo folder",
    )
    p_index.add_argument(
        "--reindex",
        action="store_true",
        help="Delete and rewrite faces for each image instead of skipping",
    )

    # --- search ---
    p_search = sub.add_parser("search", help="Find photos matching a reference face")
    p_search.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Reference photo of the person to find",
    )
    p_search.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum cosine similarity (default {DEFAULT_THRESHOLD})",
    )
    p_search.add_argument(
        "--face-index",
        type=int,
        default=None,
        help="Which detected face in the reference image to use (default: largest)",
    )
    p_search.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max face rows to fetch from pgvector before filtering (default 50)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Parse CLI args and dispatch to index or search.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        return cmd_index(args.folder, args.reindex)
    if args.command == "search":
        return cmd_search(args.image, args.threshold, args.face_index, args.limit)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
