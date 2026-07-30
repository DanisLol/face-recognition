CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS faces (
  id           BIGSERIAL PRIMARY KEY,
  image_path   TEXT NOT NULL,
  image_name   TEXT NOT NULL,
  face_index   INT  NOT NULL,
  bbox_x       INT,
  bbox_y       INT,
  bbox_w       INT,
  bbox_h       INT,
  embedding    vector(512) NOT NULL,
  created_at   TIMESTAMPTZ DEFAULT now(),
  UNIQUE (image_path, face_index)
);