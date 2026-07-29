import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

MODEL = "/Users/danielwang/Documents/Projects/face-recognition/blaze_face_full_range.tflite"
#we use full range instead of short range to detect photos where faces are far away

IMAGE = "/Users/danielwang/Documents/Projects/face-recognition/the_office.jpg"



options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.3, # minimum confidence score for the face detectection to be
                                #considered succesful
    min_supression_threshold = 0.3,
)

with vision.FaceDetector.create_from_options(options) as detector:
    mp_image = mp.Image.create_from_file(IMAGE)
    result = detector.detect(mp_image)

img = cv2.imread(IMAGE)
h, w, _ = img.shape

for i, detection in enumerate(result.detections):
    box = detection.bounding_box
    x, y = box.origin_x, box.origin_y
    bw, bh = box.width, box.height
    # clamp to image bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(f"stored-faces/{i}.jpg", crop)

print(f"Found {len(result.detections)} faces")