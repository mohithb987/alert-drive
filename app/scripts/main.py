import os
import sys
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import torch
from detect_and_draw import detect_and_draw_faces, get_confidence
from wheel_detection import SimpleCNN
from ultralytics import YOLO
from outbound_call import make_outbound_call
import threading

# Ensure paths.py can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from paths import MASK_MODEL_PATH, SUNGLASSES_MODEL_PATH, SIMPLE_CNN_MODEL_PATH, SHAPE_PREDICTOR_PATH, VIDEO_PATH

def compute_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def compute_mar(mouth):
    A = distance.euclidean(mouth[1], mouth[7])
    B = distance.euclidean(mouth[2], mouth[6])
    C = distance.euclidean(mouth[3], mouth[5])
    D = distance.euclidean(mouth[0], mouth[4])
    return (A + B + C) / (3.0 * D)

def get_landmarks(gray, detector, predictor, bbox):
    x, y, w, h = bbox
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    landmarks = predictor(gray, rect)
    return np.array([[p.x, p.y] for p in landmarks.parts()])

def detect_occlusions(frame, mask_model, sunglasses_model):
    def detect(model):
        results = model(frame)
        boxes = results[0].boxes
        if boxes is not None:
            confidences = boxes.conf
            return any(conf > 0.5 for conf in confidences)
        return False

    masks_detected = detect(mask_model)
    sunglasses_detected = detect(sunglasses_model)

    print("Masks detected!" if masks_detected else "No masks detected.")
    print("Sunglasses detected!" if sunglasses_detected else "No sunglasses detected.")

    return masks_detected, sunglasses_detected

def check_head_bending(landmarks):
    left_eye_center = landmarks[36:42].mean(axis=0)
    right_eye_center = landmarks[42:48].mean(axis=0)
    eye_direction = right_eye_center - left_eye_center
    angle_with_horizontal = np.arctan2(eye_direction[1], eye_direction[0]) * (180 / np.pi)
    angle_with_vertical = 90 - abs(angle_with_horizontal)
    return angle_with_vertical > 20  # bending threshold

def is_drowsy(ear_list, threshold=0.3):
    below_threshold = [ear < threshold for ear in ear_list]
    return sum(below_threshold) > len(ear_list) // 2

def speak(message):
    os.system(f'say -v Shelley "{message}"')  # Works only on MAC. Check for Windows/Ubuntu

def initialize_models():
    mask_model = YOLO(MASK_MODEL_PATH)
    sunglasses_model = YOLO(SUNGLASSES_MODEL_PATH)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    model = SimpleCNN(input_shape=(1, 200, 200), num_classes=2)
    model.load_state_dict(torch.load(SIMPLE_CNN_MODEL_PATH))

    return mask_model, sunglasses_model, detector, predictor, model

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    return cap

def get_initial_bbox_and_confidence(image, model, detector):
    _, face1, face2, p1_coordinates, p2_coordinates = detect_and_draw_faces(image)

    if not p1_coordinates:
        raise ValueError("No Face Detected")

    if not p2_coordinates:
        return calculate_bbox(face1, image, model, p1_coordinates)
    else:
        return compare_faces_and_get_bbox(face1, face2, image, model, p1_coordinates, p2_coordinates)

def calculate_bbox(face, image, model, coordinates):
    region = image[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
    confidence = get_confidence(None, region, model)
    x, y, w, h = face[0], face[1], face[2] - face[0], face[3] - face[1]
    w, h, x, y = int(w * 1.1), int(h * 1.1), int(x - (w - (face[2] - face[0])) / 2), int(y - (h - (face[3] - face[1])) / 2)
    return (x, y, w, h), confidence

def compare_faces_and_get_bbox(face1, face2, image, model, p1_coordinates, p2_coordinates):
    p1_bbox, p1_confidence = calculate_bbox(face1, image, model, p1_coordinates)
    p2_bbox, p2_confidence = calculate_bbox(face2, image, model, p2_coordinates)

    if p1_confidence[0][1].item() > p1_confidence[0][0].item() or \
       (p1_confidence[0][1].item() < p1_confidence[0][0].item() and p2_confidence[0][1].item() < p2_confidence[0][0].item()):
        return p1_bbox
    else:
        return p2_bbox

def main():
    video_path = VIDEO_PATH
    cap = initialize_video_capture(video_path)

    mask_model, sunglasses_model, detector, predictor, model = initialize_models()

    ret, image = cap.read()
    initial_bbox = get_initial_bbox_and_confidence(image, model, detector)

    tracker = cv2.TrackerCSRT_create()
    tracker.init(image, initial_bbox)
    
    ear_list = []
    alarm_on = False
    drowsiness_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video.")
            break

        ret, bbox = tracker.update(frame)
        if ret:
            process_frame(frame, bbox, detector, predictor, mask_model, sunglasses_model, ear_list, alarm_on, drowsiness_count)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame, bbox, detector, predictor, mask_model, sunglasses_model, ear_list, alarm_on, drowsiness_count):
    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    masks_detected, sunglasses_detected = detect_occlusions(frame, mask_model, sunglasses_model)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    landmarks = get_landmarks(gray, detector, predictor, bbox)

    left_eye, right_eye = landmarks[36:42], landmarks[42:48]
    mouth = landmarks[48:68]

    ear_left = compute_ear(left_eye)
    ear_right = compute_ear(right_eye)
    mar = compute_mar(mouth)

    display_metrics(frame, ear_left, ear_right, mar)
    visualize_landmarks(frame, left_eye, right_eye, mouth)

    drowsiness_status = determine_drowsiness(sunglasses_detected, masks_detected, ear_left, ear_right, landmarks, alarm_on, drowsiness_count)
    if drowsiness_status == "Symptoms of drowsiness":
        drowsiness_count += 1
        if drowsiness_count >= 5 and not alarm_on:
            alarm_on = True
            threading.Thread(target=speak, args=("Stay alert",)).start()
        elif alarm_on and drowsiness_count == 11:
            threading.Thread(target=make_outbound_call).start()

def display_metrics(frame, ear_left, ear_right, mar):
    cv2.putText(frame, f"EAR (Left): {ear_left:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"EAR (Right): {ear_right:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def visualize_landmarks(frame, left_eye, right_eye, mouth):
    for (x, y) in left_eye:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    for (x, y) in right_eye:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    for (x, y) in mouth:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

def visualize_landmarks(frame, left_eye, right_eye, mouth):
    for (x, y) in left_eye:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    for (x, y) in right_eye:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    for (x, y) in mouth:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

def determine_drowsiness(sunglasses_detected, masks_detected, ear_left, ear_right, landmarks, alarm_on, drowsiness_count):
    head_bending = check_head_bending(landmarks)
    ear_list = [ear_left, ear_right]
    
    if sunglasses_detected and masks_detected:
        if head_bending:
            return "Symptoms of drowsiness"
        else:
            return "No drowsiness detected"
    
    if sunglasses_detected:
        mouth_open_ratio = compute_mar(landmarks[48:68])
        if head_bending or mouth_open_ratio > 0.6:
            return "Symptoms of drowsiness"
        else:
            return "No drowsiness detected"
    
    if masks_detected:
        if head_bending:
            return "Symptoms of drowsiness"
        else:
            return "No drowsiness detected"

    # No occlusions detected: Check EAR
    if is_drowsy(ear_list):
        return "Symptoms of drowsiness"
    
    return "No drowsiness detected"

if __name__ == "__main__":
    main()
