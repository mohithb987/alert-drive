import cv2
import os


cascade_path = 'haarcascade_frontalface_default.xml'

if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haar cascade XML file not found at: {cascade_path}")

input_video_path = 'data/video.mp4'
output_video_path = 'output/video_with_faces.mp4'

cap = cv2.VideoCapture(input_video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_and_draw_bounding_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_faces = detect_and_draw_bounding_box(frame)
    cv2.imshow('Video with Detected Faces', frame_with_faces)
    out.write(frame_with_faces)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()