from face_detect import detect_faces

faces = detect_faces("the_office.jpg")
print(f"Found {len(faces)} faces")