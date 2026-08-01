#Arc face embedding

import numpy as np
from insightface.app import FaceAnalysis
import cv2 as cv


#insight arcface for finding identity


_APP = None #private module-level cache variable

#lazy loader + cache:
#create FaceAnalysis, call prepare and stroe it in _APP
def _get_app():
    """
    Loading InsightFace's FaceAnalysis and buffalo_l model
    """
    global _APP
    if _APP is None:
        _APP = FaceAnalysis(name="buffalo_l", providers = ["CPUExecutionProvider"])
        _APP.prepare(ctx_id=-1) #-1 is CPU, 0 is first GPU and 1 is Second GPU, etc.
    return _APP


def face_area(f):
    width = f.bbox[2] - f.bbox[0] #x2 - x1
    height = f.bbox[3] - f.bbox[1] #y2 - y1
    return width * height


def embed_face(crop: np.ndarray) -> np.ndarray:
    """
    Return L2-normalized ArcFace embedding, shape (512,), dtype float32.
    """
    #accept a crop array for the face in the image and return one vector 
    #validate crop insightface

    if crop is None or not isinstance(crop, np.ndarray):
        raise ValueError("crop must be a numpy array")
    if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError(f"crop must be HxWx3, got shape {getattr(crop, 'shape', None)}")

    pad_ratio = 0.2
    h, w = crop.shape[:2]
    top = int(h * pad_ratio)
    bottom = int(h * pad_ratio)
    left = int(w * pad_ratio)
    right = int(w * pad_ratio)
    padded = cv.copyMakeBorder(
        crop, top, bottom, left, right, cv.BORDER_REFLECT
    )
    
    faces = _get_app().get(padded)
    if faces == []:
        raise ValueError("No face found in crop for embedding")
    face = max(faces, key=face_area) #if insightface find more than one face in the padded crop, the largest box is usually the face BlazeFace provided
    
    v = face.normed_embedding.astype(np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError
    v = v/ norm
    return v






