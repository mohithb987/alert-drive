import cv2
import os

cascade_path = 'haarcascade_frontalface_default.xml'

if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haar cascade XML file not found at: {cascade_path}")

input_directory = 'images/'
output_directory = 'output_haar/'

os.makedirs(output_directory, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_and_draw_driver_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    
    largest_face_area = 0
    driver_face = None
    
    for (x, y, w, h) in faces:
        print('Face detected')
        if w * h > largest_face_area:
            largest_face_area = w * h
            driver_face = (x, y, w, h)
    
    if driver_face:
        x, y, w, h = driver_face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image

for filename in os.listdir(input_directory):
    image_path = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, f'detected_{filename}')
    print(f'detected_{filename}')
    
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    image_with_driver_face = detect_and_draw_driver_face(image)
    cv2.imwrite(output_path, image_with_driver_face)

    cv2.imshow('Image with Driver\'s Face', image_with_driver_face)
    cv2.waitKey(0)

cv2.destroyAllWindows()