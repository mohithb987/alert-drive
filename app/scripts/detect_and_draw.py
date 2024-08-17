import cv2
import os
import math
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms
from PIL import Image
from wheel_detection import SimpleCNN
from extract_roi_images import extract_frames

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from paths import TRAIN_VIDEOS_PATH, INTERMEDIATE_ROI_PATH, SAVED_MODELS_DIR_PATH, INTERMEDIATE_CLASSIFICATION_PATH

def detect_faces_sort_by_area(image):
    """
    Detects faces and returns the boxes sorted in decreasing order of their area.
    """
    img_height, img_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    results = mp_face_detection.process(image_rgb)    
    faces = []
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            xmin = int(bboxC.xmin * img_width)
            ymin = int(bboxC.ymin * img_height)
            width = int(bboxC.width * img_width)
            height = int(bboxC.height * img_height)
            xmax = xmin + width
            ymax = ymin + height
            area = width * height
            faces.append({'bbox': (xmin, ymin, xmax, ymax), 'area': area})

    faces_sorted = sorted(faces, key=lambda x: x['area'], reverse=True)
    sorted_bboxes = [face['bbox'] for face in faces_sorted]
    
    return sorted_bboxes


def detect_and_draw_faces(image):
    """
    Returns the 2 largest faces (driver and shotgun passenger) along with
    region below their faces to perform steering wheel detection.
    """
    faces_sorted = detect_faces_sort_by_area(image)
    p1_coordinates, p2_coordinates, face1, face2 = [], [], [], []
    
    for i, face in enumerate(faces_sorted):
        x, y, xmax, ymax = face
        half_image_width = image.shape[1] // 2
        
        if x < half_image_width:
            orange_box_start_x = 0
            orange_box_end_x = math.ceil(0.4 * half_image_width)
        else:
            orange_box_start_x = math.ceil(1.6 * half_image_width)
            orange_box_end_x = image.shape[1]

        orange_box_start_y = ymax + math.ceil(0.5 * (image.shape[0] - ymax))
        orange_box_end_y = image.shape[0]

        if i == 0:  # Face with the largest area
            face1 = face
            p1_coordinates = [orange_box_start_x, orange_box_start_y, orange_box_end_x, orange_box_end_y]
        elif i == 1:  # Second largest face
            face2 = face
            p2_coordinates = [orange_box_start_x, orange_box_start_y, orange_box_end_x, orange_box_end_y]
        
        if i >= 1:
            break

    return image, face1, face2, p1_coordinates, p2_coordinates


def get_confidence(idx, image_region, model):    
    """
    Returns confidence of steering wheel presence in the given image.
    """
    image_region_pil = Image.fromarray(image_region)

    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    transformed_img = transform(image_region_pil)
    model_input = torch.tensor(transformed_img.unsqueeze(0))
    confidence = model(model_input)

    print(f'Confidence of class 0: {confidence[0][0].item()}')
    print(f'Confidence of class 1: {confidence[0][1].item()}')
    
    return confidence


if __name__ == "__main__":
    model = SimpleCNN(input_shape=(1, 200, 200), num_classes=2)
    model = torch.load(os.path.join(SAVED_MODELS_DIR_PATH, 'simple_cnn_model7.pth'))

    for idx, vid_path in enumerate(os.listdir(TRAIN_VIDEOS_PATH)):
        if vid_path == '.DS_Store':
            continue

        intermediate_dir = os.path.join(INTERMEDIATE_ROI_PATH, str(idx))
        os.makedirs(intermediate_dir, exist_ok=True)
        extract_frames(os.path.join(TRAIN_VIDEOS_PATH, vid_path), intermediate_dir)

    input_directories = [d for d in os.listdir(INTERMEDIATE_ROI_PATH) if d != '.DS_Store']
    os.makedirs(INTERMEDIATE_CLASSIFICATION_PATH, exist_ok=True)

    for dir_idx, input_directory in enumerate(input_directories):
        print(f'dir_idx: {dir_idx} | Input Directory: {input_directory}')
        
        for idx, filename in enumerate(os.listdir(os.path.join(INTERMEDIATE_ROI_PATH, input_directory))):
            if filename == '.DS_Store':
                continue

            image_path = os.path.join(INTERMEDIATE_ROI_PATH, input_directory, filename)
            output_path = os.path.join(INTERMEDIATE_CLASSIFICATION_PATH, f'{dir_idx}_detected_{filename}')
            image = cv2.imread(image_path)

            if image is None:
                continue
            
            _, face1, face2, p1_coordinates, p2_coordinates = detect_and_draw_faces(image)
            
            if not p1_coordinates:
                continue

            if not p2_coordinates:
                p1_region = image[p1_coordinates[1]:p1_coordinates[3], p1_coordinates[0]:p1_coordinates[2]]
                p1_confidence = get_confidence(f'p1_{idx}', p1_region, model)

                p1_color = (0, 255, 0)
                p1_label = 'Driver'

                cv2.rectangle(image, (face1[0], face1[1]), (face1[2], face1[3]), p1_color, 2)
                cv2.putText(image, p1_label, (face1[0], face1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, p1_color, 2)
                cv2.imwrite(output_path, image)
                cv2.imshow('Image with Detected Faces and Bounding Boxes', image)
                cv2.waitKey(0)
            else:
                p1_region = image[p1_coordinates[1]:p1_coordinates[3], p1_coordinates[0]:p1_coordinates[2]]
                p1_confidence = get_confidence(f'p1_{idx}', p1_region, model)
                p2_region = image[p2_coordinates[1]:p2_coordinates[3], p2_coordinates[0]:p2_coordinates[2]]
                p2_confidence = get_confidence(f'p2_{idx}', p2_region, model)

                p1_color, p2_color = (255, 0, 0), (255, 0, 0)
                p1_label, p2_label = 'Shotgun', 'Shotgun'

                if p1_confidence[0][1].item() > p1_confidence[0][0].item() or (p1_confidence[0][1].item() < p1_confidence[0][0].item() and p2_confidence[0][1].item() < p2_confidence[0][0].item()):
                    p1_color = (0, 255, 0)
                    p1_label = 'Driver'
                else:
                    p2_color = (0, 255, 0)
                    p2_label = 'Driver'

                cv2.rectangle(image, (face1[0], face1[1]), (face1[2], face1[3]), p1_color, 2)
                cv2.putText(image, p1_label, (face1[0], face1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, p1_color, 2)
                cv2.rectangle(image, (face2[0], face2[1]), (face2[2], face2[3]), p2_color, 2)
                cv2.putText(image, p2_label, (face2[0], face2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, p2_color, 2)

                cv2.imwrite(output_path, image)
                cv2.imshow('Image with Detected Faces and Bounding Boxes', image)
                cv2.waitKey(0)

    cv2.destroyAllWindows()