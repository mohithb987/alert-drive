from detect_and_draw import detect_and_draw_faces, get_confidence
from wheel_detection import SimpleCNN

if __name__ == "__main__":
    import cv2
    import torch

    video_path = '../../input/data/videos/test/9.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    model = SimpleCNN(input_shape=(1, 200, 200), num_classes=2)
    model_path = '../../model/trained_model/simple_cnn_model7.pth'
    model = torch.load(model_path)
    
    ret, image = cap.read()
    _, face1, face2, p1_coordinates, p2_coordinates = detect_and_draw_faces(image)

    print("Face 1 Bounding Box:", face1)
    print("P1 Coordinates:", p1_coordinates)
    print("Face 2 Bounding Box:", face2)
    print("P2 Coordinates:", p2_coordinates)

    # Testing with an orange rectangle around the detected faces
    if p1_coordinates:
        print("Drawing rectangle for P1")
        cv2.rectangle(image, (face1[0], face1[1]), 
                      (face1[2], face1[3]), (0, 165, 255), 2)

    cv2.imshow("Test Image with Rectangles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    if not p1_coordinates:
        print('Error: No Face Detected')

    if not p2_coordinates:
        print('Only 1 face detected')
        p1_region = image[p1_coordinates[1]:p1_coordinates[3], p1_coordinates[0]:p1_coordinates[2]]
        p1_confidence = get_confidence(None, p1_region, model)
        x, y, w, h = face1[0], face1[1], face1[2] - face1[0], face1[3] - face1[1]
        w = int(w*1.1)
        h = int(h*1.1)
        x = int(x - (w -(face1[2] - face1[0]))/2)
        y = int(y - (h -(face1[3] - face1[1]))/2)
        initial_bbox = (x, y, w, h)

    else:
        p1_region = image[p1_coordinates[1]:p1_coordinates[3], p1_coordinates[0]:p1_coordinates[2]]
        p1_confidence = get_confidence(None, p1_region, model)
        print('### P1 Region Shape:', p1_region.shape)

        p2_region = image[p2_coordinates[1]:p2_coordinates[3], p2_coordinates[0]:p2_coordinates[2]]
        p2_confidence = get_confidence(None, p2_region, model)

        if p1_confidence[0][1].item() > p1_confidence[0][0].item() or (p1_confidence[0][1].item() < p1_confidence[0][0].item() and p2_confidence[0][1].item() < p2_confidence[0][0].item()):
            x, y, w, h = face1[0], face1[1], face1[2] - face1[0], face1[3] - face1[1]
            w = int(w*1.1)
            h = int(h*1.1)
            x = int(x - (w -(face1[2] - face1[0]))/2)
            y = int(y - (h -(face1[3] - face1[1]))/2)
            initial_bbox = (x, y, w, h)

        
        else:
            x, y, w, h = face2[0], face2[1], face2[2] - face2[0], face2[3] - face2[1]
            w = int(w*1.1)
            h = int(h*1.1)
            x = int(x - (w -(face2[2] - face2[0]))/2)
            y = int(y - (h -(face2[3] - face2[1]))/2)
            initial_bbox = (x, y, w, h)


    tracker = cv2.TrackerCSRT_create()
    tracker.init(image, initial_bbox)

    # Perform driver face tracking
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
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
    
    # detect_and_draw_faces(img)
    # driver_coords = wheel_detection()
    # sunglasses, mask = occlusions_detection(img, driver_coords)

    # if sunglasses and not mask:
    #    mor = 
    #    headpose = 
    
    # elif sunglasses and mask:
    #    headpose = 

    # else:
    #    ear = 
    #    mor =     #to issue warnings
