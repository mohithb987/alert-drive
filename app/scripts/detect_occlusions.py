import cv2
import numpy as np
from ultralytics import YOLO
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from paths import SAVED_MODELS_DIR_PATH, INTERMEDIATE_ROI_PATH, INTERMEDIATE_CLASSIFICATION_PATH

# Load models using paths from paths.py
model_sunglasses = YOLO(os.path.join(SAVED_MODELS_DIR_PATH, 'yolov8_sunglasses_model.pt'))
model_masks = YOLO(os.path.join(SAVED_MODELS_DIR_PATH, 'yolov8_mask_model.pt'))

def draw_bounding_boxes(img, results, label):
    """
    Draws bounding boxes on the image for detected objects with confidence greater than 0.7.
    """
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        confidences = result.boxes.conf.numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]
            
            if confidence > 0.7:
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def detect_occlusions(image_path):
    """
    Detects sunglasses and masks in the image and draws bounding boxes around them.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results_sunglasses = model_sunglasses(img_rgb)[0]
    results_masks = model_masks(img_rgb)[0]

    draw_bounding_boxes(img, results_sunglasses, 'Sunglasses')
    draw_bounding_boxes(img, results_masks, 'Mask')

    # Save output using path from paths.py
    output_path = os.path.join(INTERMEDIATE_CLASSIFICATION_PATH, 'new_test.jpg')
    cv2.imwrite(output_path, img)
    print(f"Output image saved to {output_path}")

# Example usage
image_path = os.path.join(INTERMEDIATE_ROI_PATH, '29', 'copy.jpg')
detect_occlusions(image_path)