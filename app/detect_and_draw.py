import cv2
import os
from mtcnn import MTCNN
import numpy as np


def detect_and_draw_faces(image):
    import cv2
    from mtcnn import MTCNN

    detector = MTCNN()
    faces = detector.detect_faces(image)
    faces_sorted = sorted(faces, key=lambda x: x['box'][2] * x['box'][3], reverse=True) # sort based on area occupied by the face

    p1_coordinates = []
    p2_coordinates = []
    face1 = []
    face2 = []
    for i, face in enumerate(faces_sorted):
        (x, y, w, h) = face['box']
        half_image_width = image.shape[1] // 2
        if x < half_image_width:
            orange_box_start_x = 0 
            orange_box_end_x = half_image_width
        else:
            orange_box_start_x = half_image_width
            orange_box_end_x = image.shape[1]

        if i == 0:  # Face with largest area (could be the driver)
            color = (0, 255, 0) 
            # label = 'Driver?' 
            # cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Capture area below the face to perform steering wheel detection 
            face1 = [x, y, x+w, y+h]
            orange_box_start_y = y + h
            orange_box_end_y = image.shape[0]
            
            # cv2.rectangle(image, (orange_box_start_x, orange_box_start_y), (orange_box_end_x, orange_box_end_y), (0, 165, 255), 2)
            p1_coordinates = [orange_box_start_x, orange_box_start_y, orange_box_end_x, orange_box_end_y]

        elif i == 1:  # Second largest face (could be the shotgun passenger)
            color = (255, 0, 0)
            # label = 'Shotgun Passenger?'
            # cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Capture area below the face to perform steering wheel detection
            face2 = [x, y, x+w, y+h]
            orange_box_start_y = y + h
            orange_box_end_y = image.shape[0]
            
            # cv2.rectangle(image, (orange_box_start_x, orange_box_start_y), (orange_box_end_x, orange_box_end_y), (0, 165, 255), 2)
            p2_coordinates = [orange_box_start_x, orange_box_start_y, orange_box_end_x, orange_box_end_y]
        else:
            break

    return image, face1, face2, p1_coordinates, p2_coordinates


def predict_confidence(image_region, model):    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])

    transformed_img = transform(image_region).unsqueeze(0)
    confidence = model(transformed_img)
    print('Confidence of class 0:',  confidence[0][0].item())
    print('Confidence of Class 1:',  confidence[0][1].item())
    
    return [confidence[0][0].item(), confidence[0][1].item()]    


if __name__ == "__main__":
    import cv2
    import os
    import torch
    from wheel_detection import SimpleCNN

    input_directory = '../input/test/'
    output_directory = '../output/faces_classification/'

    os.makedirs(output_directory, exist_ok=True)
    model = SimpleCNN(input_shape=(3, 256, 256), num_classes=2)
    model_path = '../model/trained_model/simple_cnn_model4.pth'
    model = torch.load(model_path)

    for filename in os.listdir(input_directory):
        image_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f'detected_{filename}')
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        _, face1, face2, p1_coordinates, p2_coordinates = detect_and_draw_faces(image)
        
        if p1_coordinates and p2_coordinates:
            p1_region = image[p1_coordinates[1]:p1_coordinates[3], p1_coordinates[0]:p1_coordinates[2]]
            p1_confidence = predict_confidence(p1_region, model)
            # p1_label = f'Confidence: Class 0 {p1_confidence[0]:.2f}, Class 1 {p1_confidence[1]:.2f}'
            # cv2.putText(image_with_bounding_boxes, p1_label, (p1_coordinates[0], p1_coordinates[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            cv2.imshow(f'Region 1', p1_region)

            p2_region = image[p2_coordinates[1]:p2_coordinates[3], p2_coordinates[0]:p2_coordinates[2]]
            p2_confidence = predict_confidence(p2_region, model)
            # p2_label = f'Confidence: Class 0 {p2_confidence[0]:.2f}, Class 1 {p2_confidence[1]:.2f}'
            # cv2.putText(image_with_bounding_boxes, p2_label, (p2_coordinates[0], p2_coordinates[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            cv2.imshow(f'Region 2:', p2_region)
            p1_color = (255, 0, 0)
            p2_color = (255, 0, 0)
            p1_label = 'Shotgun'
            p2_label = 'Shotgun'
            
            if p1_confidence[1]>p1_confidence[0]:
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