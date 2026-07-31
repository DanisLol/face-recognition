from dataclasses import dataclass
import numpy as np
import cv2 as cv
from pathlib import Path

#BlazeFace configuration
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "blaze_face_full_range.tflite"

@dataclass
class DetectedFace:
    """One face found in an image."""
    face_index: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    crop: np.ndarray  # BGR crop from OpenCV, used later by face_embed.py

def detect_faces(image_path: str, model_path: str | Path | None = None) -> list[DetectedFace]:
    """
    Detect faces in an image using MediaPipe BlazeFace Far-Range
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
        model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"BalzeFace model not found")
    
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    else:
        h, w, _ = img.shape #clamp coordinates to make box stay inside the photo and not leak outside
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.3, #ignore weak maybe faces
            min_suppression_threshold=0.3, #merge overlapping duplicated boxes
        )

        with vision.FaceDetector.create_from_options(options) as detector:
            mp_image = mp.Image.create_from_file(str(image_path))
            result = detector.detect(mp_image) #return boxes

        faces: list[DetectedFace] = []

        detections = result.detections

        for i, detection in enumerate(detections):
            box = detection.bounding_box
            x, y = box.origin_x, box.origin_y
            bw, bh = box.width, box.height

            # clamp to image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)

            # skip empty/invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]

            faces.append(
                DetectedFace(
                    face_index=i,
                    bbox_x=x1,
                    bbox_y=y1,
                    bbox_w=x2 - x1,
                    bbox_h=y2 - y1,
                    crop=crop,
                )
            )

        return faces