import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import torch
from detect_and_draw import detect_and_draw_faces, get_confidence
from wheel_detection import SimpleCNN
from ultralytics import YOLO
from outbound_call import make_outbound_call
import os
import threading

def compute_ear(eye): #reference: https://www.mdpi.com/2313-433X/9/5/91
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    C = distance.euclidean(eye[0],eye[3])
    ear = (A+B)/(2.0*C)
    return ear

def compute_mar(mouth):
    A = distance.euclidean(mouth[1],mouth[7])
    B = distance.euclidean(mouth[2],mouth[6])
    C = distance.euclidean(mouth[3],mouth[5])
    D = distance.euclidean(mouth[0],mouth[4])
    mar = (A+B+C)/(3.0*D)
    return mar

def get_landmarks(gray, detector, predictor, bbox):
    x, y, w, h = bbox
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    landmarks = predictor(gray, rect)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
    return landmarks

def detect_occlusions(frame, mask_model, sunglasses_model):
    results_mask = mask_model(frame)
    boxes = results_mask[0].boxes
    if boxes is not None:
        boxes_data = boxes.xyxy
        confidences = boxes.conf
        masks_detected = any(conf>0.5 for conf in confidences)
        if masks_detected:
            print("Masks detected!")
        else:
            print("No masks detected.")
    else:
        print("No boxes detected.")
    
    results_sunglasses = sunglasses_model(frame)
    boxes = results_sunglasses[0].boxes
    if boxes is not None:
        boxes_data = boxes.xyxy
        confidences = boxes.conf
        sunglasses_detected = any(conf>0.5 for conf in confidences)
        if sunglasses_detected:
            print("Sunglasses detected!")
        else:
            print("No sunglasses detected.")
    else:
        print("No boxes detected.")
    
    return masks_detected, sunglasses_detected


def check_head_bending(landmarks):
    """
    Check if the head is bending based on the facial landmarks.
    """
    left_eye_center = landmarks[36:42].mean(axis=0)
    right_eye_center = landmarks[42:48].mean(axis=0)
    eye_direction = right_eye_center - left_eye_center
    angle_with_horizontal = np.arctan2(eye_direction[1], eye_direction[0])*(180/np.pi)
    angle_with_vertical = 90-abs(angle_with_horizontal)
    bending_threshold = 20
    if angle_with_vertical>bending_threshold:
        return True
    else:
        return False

def is_drowsy(ear_list, threshold=0.3): #reference: https://www.mdpi.com/2079-9292/11/19/3183
    below_threshold = [ear<threshold for ear in ear_list]
    return sum(below_threshold)>len(ear_list)//2


def speak(message):
    os.system(f'say -v Shelley "{message}"') #NOTE: Works only on MAC. Check for Windows/Ubuntu

if __name__ == "__main__":
    # video_path = '../../input/data/videos/test/9.mp4'
    video_path = '../../input/data/videos/test/new.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video")
        exit()

    import torch
    mask_model_path = '../../model/trained_model/yolov8_mask_model.pt'
    sunglasses_model_path = '../../model/trained_model/yolov8_sunglasses_model.pt'
    mask_model = YOLO(mask_model_path)
    sunglasses_model = YOLO(sunglasses_model_path)

    # Load facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Load drowsiness detection model
    model = SimpleCNN(input_shape=(1, 200, 200), num_classes=2)
    model_path = '../../model/trained_model/simple_cnn_model7.pth'
    model = torch.load(model_path)

    ret, image = cap.read()
    _, face1, face2, p1_coordinates, p2_coordinates = detect_and_draw_faces(image)

    if not p1_coordinates:
        print('No Face Detected')
        exit()

    if not p2_coordinates:
        print('Only 1 face detected')
        p1_region = image[p1_coordinates[1]:p1_coordinates[3], p1_coordinates[0]:p1_coordinates[2]]
        p1_confidence = get_confidence(None, p1_region, model)
        x, y, w, h = face1[0], face1[1], face1[2] - face1[0], face1[3] - face1[1]
        w = int(w*1.1)
        h = int(h*1.1)
        x = int(x-(w-(face1[2]-face1[0]))/2)
        y = int(y-(h-(face1[3]-face1[1]))/2)
        initial_bbox = (x, y, w, h)
    else:
        p1_region = image[p1_coordinates[1]:p1_coordinates[3], p1_coordinates[0]:p1_coordinates[2]]
        p1_confidence = get_confidence(None, p1_region, model)

        p2_region = image[p2_coordinates[1]:p2_coordinates[3], p2_coordinates[0]:p2_coordinates[2]]
        p2_confidence = get_confidence(None, p2_region, model)

        if p1_confidence[0][1].item() > p1_confidence[0][0].item() or (p1_confidence[0][1].item() < p1_confidence[0][0].item() and p2_confidence[0][1].item() < p2_confidence[0][0].item()):
            x, y, w, h = face1[0], face1[1], face1[2]-face1[0], face1[3]-face1[1]
            w = int(w*1.1)
            h = int(h*1.1)
            x = int(x-(w-(face1[2]-face1[0]))/2)
            y = int(y-(h-(face1[3]-face1[1]))/2)
            initial_bbox = (x, y, w, h)
        else:
            x, y, w, h = face2[0], face2[1], face2[2]-face2[0], face2[3]-face2[1]
            w = int(w*1.1)
            h = int(h*1.1)
            x = int(x-(w-(face2[2]-face2[0]))/2)
            y = int(y-(h-(face2[3]-face2[1]))/2)
            initial_bbox = (x, y, w, h)

    tracker = cv2.TrackerCSRT_create() 
    tracker.init(image, initial_bbox)
    ear_list = []
    alarm_on = False
    alarm_time = 0
    drowsiness_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video.")
            break
        
        ret, bbox = tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

            masks_detected, sunglasses_detected = detect_occlusions(frame, mask_model, sunglasses_model)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = get_landmarks(gray, detector, predictor, bbox)
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            mouth = landmarks[48:68]

            ear_left = compute_ear(left_eye)
            ear_right = compute_ear(right_eye)
            mar = compute_mar(mouth)

            cv2.putText(frame, f"EAR (Left): {ear_left:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"EAR (Right): {ear_right:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            #Scale this to enlarge landmark screens
            eye_scale = 7
            mouth_scale = 7

            left_eye_center = left_eye.mean(axis=0).astype("int")
            left_eye_region = frame[left_eye_center[1]-10:left_eye_center[1]+10, 
                                    left_eye_center[0]-20:left_eye_center[0]+20]
            left_eye_resized = cv2.resize(left_eye_region, None, fx=eye_scale, fy=eye_scale, interpolation=cv2.INTER_LINEAR)
            for (x, y) in left_eye - [left_eye_center[0]-20, left_eye_center[1]-10]:
                cv2.circle(left_eye_resized, (x*eye_scale, y*eye_scale), 2, (0, 0, 255), -1)
                cv2.imshow("Left Eye", left_eye_resized)
            right_eye_center = right_eye.mean(axis=0).astype("int")
            right_eye_region = frame[right_eye_center[1]-10:right_eye_center[1]+10, 
                                    right_eye_center[0]-20:right_eye_center[0]+20]
            right_eye_resized = cv2.resize(right_eye_region, None, fx=eye_scale, fy=eye_scale, interpolation=cv2.INTER_LINEAR)
            for (x, y) in right_eye - [right_eye_center[0]-20, right_eye_center[1]-10]:
                cv2.circle(right_eye_resized, (x*eye_scale, y*eye_scale), 2, (0, 0, 255), -1)
            cv2.imshow("Right Eye", right_eye_resized)

            mouth_center = mouth.mean(axis=0).astype("int")
            mouth_region = frame[mouth_center[1]-10 : mouth_center[1]+10, mouth_center[0]-20 : mouth_center[0]+20]
            mouth_resized = cv2.resize(mouth_region, None, fx=mouth_scale, fy=mouth_scale, interpolation=cv2.INTER_LINEAR)
            for (x, y) in mouth-[mouth_center[0]-20, mouth_center[1]-10]:
                cv2.circle(mouth_resized, (x * mouth_scale, y * mouth_scale), 2, (0, 0, 255), -1)
            cv2.imshow("Mouth", mouth_resized)

            # Determine drowsiness status
            if sunglasses_detected:
                cv2.putText(frame, "Sunglasses Detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                drowsiness_status = "Unknown due to occlusion"
            if masks_detected:
                cv2.putText(frame, "Mask Detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                drowsiness_status = "Unknown due to occlusion"
            if check_head_bending(landmarks):
                cv2.putText(frame, "Head Bending Detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                drowsiness_status = "Unknown due to Head bending"
            if not sunglasses_detected:
                if ear_left<0.3 and ear_right<0.3:      # <!Update as needed>
                    drowsiness_status = "Drowsy"
                    drowsiness_count += 1

                    if not alarm_on and drowsiness_count>=5:
                        cv2.putText(frame, "Drowsiness Detected! Beep Alarm!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        drowsiness_status = "Symptoms of drowsiness"
                        alarm_on = True
                        threading.Thread(target=speak, args=("Stay alert",)).start()   #prevent main thread from being blocked by sys.
                    elif alarm_on and drowsiness_count==11:
                        threading.Thread(target=speak, args=("Please stay alert. You are showing signs of drowsiness",)).start()
                    elif alarm_on and drowsiness_count==15:
                        threading.Thread(target=speak, args=("Calling your close contact",)).start()
                        drowsiness_status = "Drowsy"
                        cv2.putText(frame, "Drowsiness Persistent! Making Call!", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        make_outbound_call()
                else:
                    drowsiness_status = "Alert"

            cv2.putText(frame, f"Drowsiness Status: {drowsiness_status}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()