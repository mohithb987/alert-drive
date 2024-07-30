import cv2
import numpy as np
from ultralytics import YOLO

model_sunglasses = YOLO('../../model/trained_model/yolov8_sunglasses_model.pt')
model_masks = YOLO('../../model/trained_model/yolov8_mask_model.pt')

def draw_bounding_boxes(img, results, label):
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
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results_sunglasses = model_sunglasses(img_rgb)
    results_sunglasses = results_sunglasses[0]

    results_masks = model_masks(img_rgb)
    results_masks = results_masks[0]

    draw_bounding_boxes(img, results_sunglasses, 'Sunglasses')
    draw_bounding_boxes(img, results_masks, 'Mask')

    output_path = '../../output/occlusions/new_test.jpg'
    cv2.imwrite(output_path, img)
    print(f"Output image saved to {output_path}")

image_path = '../../input/data/test_images_2/29/copy.jpg'
detect_occlusions(image_path)