import os
import sys
import cv2
from detect_and_draw import detect_and_draw_faces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from paths import TRAIN_VIDEOS_PATH, INTERMEDIATE_ROI_PATH

def extract_frames(video_path, output_dir):
    """
    Extract frames from a video file and save them as images in the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    print(f"Video frame rate (FPS): {fps}")

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1
        frame_count += 1

    video_capture.release()
    print(f"Extracted {saved_frame_count} frames from {video_path} to {output_dir} \n\n")

if __name__ == "__main__":
    input_videos_path = TRAIN_VIDEOS_PATH

    for idx, vid_path in enumerate(os.listdir(input_videos_path)):
        video_full_path = os.path.join(input_videos_path, vid_path)
        print(f'### Extracting FRAMES from video in path: {video_full_path}')
        
        train_images_dir = os.path.join(INTERMEDIATE_ROI_PATH, f'train_images_{idx}')
        extract_frames(video_full_path, train_images_dir)

        driver_rois_dir = os.path.join(train_images_dir, 'driver')
        shotgun_rois_dir = os.path.join(train_images_dir, 'shotgun')
        
        os.makedirs(driver_rois_dir, exist_ok=True)
        os.makedirs(shotgun_rois_dir, exist_ok=True)

        for filename in os.listdir(train_images_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(train_images_dir, filename)
                print(f"######## Image Frame Path for classification: {image_path}")
                
                img = cv2.imread(image_path)
                marked_img, f1, f2, p1, p2 = detect_and_draw_faces(img)
                
                if p1 and p2:
                    d_roi = marked_img[p1[1]:p1[3], p1[0]:p1[2]]
                    s_roi = marked_img[p2[1]:p2[3], p2[0]:p2[2]]
                    
                    if d_roi.size > 0 and s_roi.size > 0:
                        d_roi_path = os.path.join(driver_rois_dir, f"driver_{filename}")
                        s_roi_path = os.path.join(shotgun_rois_dir, f"shotgun_{filename}")
                        cv2.imwrite(d_roi_path, d_roi)
                        cv2.imwrite(s_roi_path, s_roi)
                    else:
                        print(f"Empty ROI detected for {filename}")
                else:
                    print(f"Invalid coordinates for {filename}")