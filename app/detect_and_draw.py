import cv2
import os
from mtcnn import MTCNN
import numpy as np
from tensorflow.keras.models import load_model


def detect_and_draw_faces(image):
    import cv2
    from mtcnn import MTCNN

    detector = MTCNN()
    faces = detector.detect_faces(image)
    faces_sorted = sorted(faces, key=lambda x: x['box'][2] * x['box'][3], reverse=True) # sort based on area occupied by the face

    p1_coordinates = []
    p2_coordinates = []

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
            label = 'Driver?' 
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Capture area below the face to perform steering wheel detection 
            orange_box_start_y = y + h
            orange_box_end_y = image.shape[0]
            
            cv2.rectangle(image, (orange_box_start_x, orange_box_start_y), (orange_box_end_x, orange_box_end_y), (0, 165, 255), 2)
            p1_coordinates = [orange_box_start_x, orange_box_start_y, orange_box_end_x, orange_box_end_y]

        elif i == 1:  # Second largest face (could be the shotgun passenger)
            color = (255, 0, 0)
            label = 'Shotgun Passenger?'
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Capture area below the face to perform steering wheel detection
            orange_box_start_y = y + h
            orange_box_end_y = image.shape[0]
            
            cv2.rectangle(image, (orange_box_start_x, orange_box_start_y), (orange_box_end_x, orange_box_end_y), (0, 165, 255), 2)
            p2_coordinates = [orange_box_start_x, orange_box_start_y, orange_box_end_x, orange_box_end_y]
        else:
            break

    return image, p1_coordinates, p2_coordinates


def predict_confidence(image_region, model):
    preprocessed_image = image_region.astype('float32') / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    confidence = model.predict(preprocessed_image)[0]
    print('Confidence of Class 0:',  confidence[0])
    print('Confidence of Class 1:',  confidence[1])
    
    return confidence



if __name__ == "__main__":
    import cv2
    import os
    from tensorflow.keras.models import load_model

    input_directory = '../input/data/images/train/'
    output_directory = '../output/faces_classification/'

    os.makedirs(output_directory, exist_ok=True)

    model = load_model('../model/trained_model/saved_model.h5')

    for filename in os.listdir(input_directory):
        image_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f'detected_{filename}')
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        image_with_bounding_boxes, p1_coordinates, p2_coordinates = detect_and_draw_faces(image)
        
        if p1_coordinates:
            p1_region = image[p1_coordinates[1]:p1_coordinates[3], p1_coordinates[0]:p1_coordinates[2]]
            p1_confidence = predict_confidence(p1_region, model)
            p1_label = f'Confidence: Class 0 {p1_confidence[0]:.2f}, Class 1 {p1_confidence[1]:.2f}'
            cv2.putText(image_with_bounding_boxes, p1_label, (p1_coordinates[0], p1_coordinates[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            cv2.imshow(f'Region 1: {p1_label}', p1_region)

        if p2_coordinates:
            p2_region = image[p2_coordinates[1]:p2_coordinates[3], p2_coordinates[0]:p2_coordinates[2]]
            p2_confidence = predict_confidence(p2_region, model)
            p2_label = f'Confidence: Class 0 {p2_confidence[0]:.2f}, Class 1 {p2_confidence[1]:.2f}'
            cv2.putText(image_with_bounding_boxes, p2_label, (p2_coordinates[0], p2_coordinates[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            cv2.imshow(f'Region 2: {p2_label}', p2_region)

        cv2.imwrite(output_path, image_with_bounding_boxes)
        cv2.imshow('Image with Detected Faces and Bounding Boxes', image_with_bounding_boxes)
        cv2.waitKey(0)

    cv2.destroyAllWindows()