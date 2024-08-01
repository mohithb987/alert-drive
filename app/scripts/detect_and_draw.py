import cv2
import os
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_faces_sort_by_area(image):
    img_height, img_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(image_rgb)    
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            xmin, ymin, width, height = int(bboxC.xmin*img_width), int(bboxC.ymin*img_height), int(bboxC.width*img_width), int(bboxC.height*img_height)
            xmax, ymax = xmin+width, ymin+height
            area = width*height
            faces.append({'bbox': (xmin, ymin, xmax, ymax), 'area': area})

    faces_sorted = sorted(faces, key=lambda x: x['area'], reverse=True)
    sorted_bboxes = [face['bbox'] for face in faces_sorted]
    
    return sorted_bboxes


def detect_and_draw_faces(image):
    import cv2
    import math
    faces_sorted = detect_faces_sort_by_area(image)
    p1_coordinates = []
    p2_coordinates = []
    face1 = []
    face2 = []
    for i, face in enumerate(faces_sorted):
        (x, y, xmax, ymax) = face
        half_image_width = image.shape[1]//2
        
        if x < half_image_width:
            orange_box_start_x = 0
            orange_box_end_x =math.ceil(0.4*half_image_width)            # !!!!! Updated to reduce ROI area !!!!!!
        else:
            orange_box_start_x = math.ceil(1.6*half_image_width)          # !!!!! Updated to reduce ROI area !!!!!!
            orange_box_end_x = image.shape[1]

        if i == 0:  # Face with largest area (could be the driver)
            color = (0, 255, 0)
            # label = 'Driver?'
            # cv2.rectangle(image, (x, y), (xmax, ymax), color, 2)
            # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Capture area below the face to perform steering wheel detection
            face1 = [x, y, xmax, ymax]
            orange_box_start_y = ymax + math.ceil(0.5*(image.shape[0] - ymax))          # !!!!! Updated to reduce ROI area !!!!!!
            orange_box_end_y = image.shape[0]
            
            # cv2.rectangle(image, (orange_box_start_x, orange_box_start_y), (orange_box_end_x, orange_box_end_y), (0, 165, 255), 2)
            p1_coordinates = [orange_box_start_x, orange_box_start_y, orange_box_end_x, orange_box_end_y]

        elif i == 1:  # Second largest face (could be the shotgun passenger)
            color = (255, 0, 0)
            # label = 'Shotgun Passenger?'
            # cv2.rectangle(image, (x, y), (xmax, ymax), color, 2)
            # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Capturing area below the face to perform steering wheel detection
            face2 = [x, y, xmax, ymax]
            orange_box_start_y = ymax + math.ceil(0.5*(image.shape[0] - ymax))          # !!!!! Updated to reduce ROI area !!!!!!
            orange_box_end_y = image.shape[0]
            
            # cv2.rectangle(image, (orange_box_start_x, orange_box_start_y), (orange_box_end_x, orange_box_end_y), (0, 165, 255), 2)
            p2_coordinates = [orange_box_start_x, orange_box_start_y, orange_box_end_x, orange_box_end_y]
        
        if i >= 1:
            break

    return image, face1, face2, p1_coordinates, p2_coordinates

def predict_confidence(idx, image_region, model):    
    from torchvision import transforms
    from PIL import Image, ImageFilter
    import cv2

    image_region_pil = Image.fromarray(image_region)

    class OverlayCannyEdges:
        def __init__(self, low_threshold=50, high_threshold=150):
            self.low_threshold = low_threshold
            self.high_threshold = high_threshold

        def __call__(self, img):
            # Convert PIL image to NumPy array
            img_np = np.array(img)
            # Apply Canny edge detection
            edges = cv2.Canny(img_np, self.low_threshold, self.high_threshold)
            # Overlay edges on the original image
            overlay = np.maximum(img_np, edges)
            # Convert back to PIL Image
            return Image.fromarray(overlay)

    transform = transforms.Compose([
        transforms.Resize((200, 200)),  # Resize to desired dimensions
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        OverlayCannyEdges(low_threshold=50, high_threshold=150),  # Apply and overlay Canny edge detection
        transforms.ToTensor(),  # Convert PIL Image to tensor
    ])

    transformed_img = transform(image_region_pil)
    numpy_img = (transformed_img.permute(1, 2, 0).numpy()*255).astype(np.uint8)
    bgr_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join('../../input/data/steering_wheel/output_augmented_images', f'_{idx}.jpg'), bgr_img)

    transformed_imgs = []
    transformed_imgs.append(transformed_img)
    X_test = torch.stack(transformed_imgs)
    model_input = torch.tensor(X_test)
    confidence = model(model_input)
    print('Confidence of class 0:',  confidence[0][0].item())
    print('Confidence of Class 1:',  confidence[0][1].item())
    
    return confidence


if __name__ == "__main__":
    import cv2
    import os
    import torch
    from wheel_detection import SimpleCNN
    from extract_roi_images import extract_frames
    # input_directories = [d for d in os.listdir('../../input/data/test_images/') ]
    # output_directory = '../../output/faces_classification/'
    # os.makedirs(output_directory, exist_ok=True)

    input_videos_path = '../../input/data/videos/all'

    for idx, vid_path in enumerate(os.listdir(input_videos_path)):
        if vid_path == '.DS_Store':
                continue
        print(f'### Extracting FRAMES from video in path: {os.path.join(input_videos_path,vid_path)}')
        test_images_dir = f'../../input/data/test_images_2/{idx}'
        os.makedirs(test_images_dir, exist_ok=True)
        extract_frames(os.path.join(input_videos_path,vid_path), test_images_dir)

    model = SimpleCNN(input_shape=(1, 200, 200), num_classes=2)
    model_path = '../../model/trained_model/simple_cnn_model7.pth'
    model = torch.load(model_path)
    input_directories = [d for d in os.listdir('../../input/data/test_images_2/') ]
    output_directory = '../../output/faces_classification_2/'
    os.makedirs(output_directory, exist_ok=True)
    for dir_idx, input_directory in enumerate(input_directories):
        print(f'dir_idx: {dir_idx} |  Input Directory: {input_directory}')
        if input_directory == '.DS_Store':
                continue
        for idx, filename in enumerate(os.listdir(os.path.join(f'../../input/data/test_images_2/',input_directory))):
            if filename == '.DS_Store':
                continue
            image_path = os.path.join('../../input/data/test_images_2/',input_directory, filename)
            output_path = os.path.join(output_directory,f'{dir_idx}_detected_{filename}')
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            _, face1, face2, p1_coordinates, p2_coordinates = detect_and_draw_faces(image)
            
            if not p1_coordinates:
                continue

            if not p2_coordinates:
                p1_region = image[p1_coordinates[1]:p1_coordinates[3], p1_coordinates[0]:p1_coordinates[2]]
                p1_confidence = predict_confidence(f'p1_{idx}', p1_region, model)

                p1_color = (0, 255, 0)  # Green color for 'Driver'
                p1_label = 'Driver'

                cv2.rectangle(image, (face1[0], face1[1]), (face1[2], face1[3]), p1_color, 2)
                cv2.putText(image, p1_label, (face1[0], face1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, p1_color, 2)
                cv2.imwrite(output_path, image)
                cv2.imshow('Image with Detected Faces and Bounding Boxes', image)
                cv2.waitKey(0)
            else:
                p1_region = image[p1_coordinates[1]:p1_coordinates[3], p1_coordinates[0]:p1_coordinates[2]]
                p1_confidence = predict_confidence(f'p1_{idx}', p1_region, model)
                # p1_label = f'Confidence: Class 0 {p1_confidence[0]:.2f}, Class 1 {p1_confidence[1]:.2f}'
                # cv2.putText(image_with_bounding_boxes, p1_label, (p1_coordinates[0], p1_coordinates[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                print('### P1 Region Shape:', p1_region.shape)
                cv2.imshow(f'Region 1', p1_region)

                p2_region = image[p2_coordinates[1]:p2_coordinates[3], p2_coordinates[0]:p2_coordinates[2]]
                p2_confidence = predict_confidence(f'p2_{idx}', p2_region, model)
                # p2_label = f'Confidence: Class 0 {p2_confidence[0]:.2f}, Class 1 {p2_confidence[1]:.2f}'
                # cv2.putText(image_with_bounding_boxes, p2_label, (p2_coordinates[0], p2_coordinates[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                cv2.imshow(f'Region 2:', p2_region)
                p1_color = (255, 0, 0)
                p2_color = (255, 0, 0)
                p1_label = 'Shotgun'
                p2_label = 'Shotgun'
                


                if p1_confidence[0][1].item() > p1_confidence[0][0].item() or (p1_confidence[0][1].item() < p1_confidence[0][0].item() and p2_confidence[0][1].item() < p2_confidence[0][0].item()):
                    p1_color = (0, 255, 0) 
                    p1_label = 'Driver'
                
                else:
                    p2_color = (0, 255, 0) 
                    p2_label = 'Driver'

                cv2.rectangle(image, (face1[0], face1[1]), (face1[2], face1[3]), p1_color, 2)
                cv2.putText(image, p1_label, (face1[0], face1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, p1_color, 2)

                cv2.rectangle(image, (face2[0], face2[1]), (face2[2], face2[3]), p2_color, 2)
                cv2.putText(image, p2_label, (face2[0], face2[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, p2_color, 2)


                cv2.imwrite(output_path, image)
                cv2.imshow('Image with Detected Faces and Bounding Boxes', image)
                cv2.waitKey(0)

    cv2.destroyAllWindows()