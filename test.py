import cv2
import os
from mtcnn import MTCNN

input_directory = 'images/'
output_directory = 'output_face_wheel/'

os.makedirs(output_directory, exist_ok=True)

detector = MTCNN()

def detect_and_draw_faces(image):
    faces = detector.detect_faces(image)
    faces_sorted = sorted(faces, key=lambda x: x['box'][2] * x['box'][3], reverse=True) # sort based on area occupied by the face

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
            
            # capture area below the face to perform steering wheel detection 
            orange_box_start_y = y + h
            orange_box_end_y = image.shape[0]
            
            cv2.rectangle(image, (orange_box_start_x, orange_box_start_y), (orange_box_end_x, orange_box_end_y), (0, 165, 255), 2)

        elif i == 1:  # Second largest face (could be the shotgun passenger)
            color = (255, 0, 0)
            label = 'Shotgun Passenger?'
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # capture area below the face to perform steering wheel detection
            orange_box_start_y = y + h
            orange_box_end_y = image.shape[0]
            
            cv2.rectangle(image, (orange_box_start_x, orange_box_start_y), (orange_box_end_x, orange_box_end_y), (0, 165, 255), 2)

        else:
            break

    return image

for filename in os.listdir(input_directory):
    image_path = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, f'detected_{filename}')
    
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    image_with_bounding_boxes = detect_and_draw_faces(image)
    cv2.imwrite(output_path, image_with_bounding_boxes)
    cv2.imshow('Image with Detected Faces and Bounding Boxes', image_with_bounding_boxes)
    cv2.waitKey(0)

cv2.destroyAllWindows()