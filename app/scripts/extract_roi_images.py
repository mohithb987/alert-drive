import cv2
import os

def extract_frames(video_path, output_dir):
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
        saved_frame_count+= 1

        frame_count+= 1

    video_capture.release()
    print(f"Extracted {saved_frame_count} frames from {video_path} to {output_dir} \n\n")


# video_path = '../../input/data/videos/train/7.mp4'
# output_dir = '../../input/data/images/train'
# extract_frames(video_path, output_dir)


if __name__ == "__main__":
    import os
    import sys
    import cv2
    from detect_and_draw import detect_and_draw_faces

    input_videos_path = '../../input/data/videos/new'

    for idx, vid_path in enumerate(os.listdir(input_videos_path)):
        print(f'### Extracting FRAMES from video in path: {os.path.join(input_videos_path,vid_path)}')
        train_images_dir = f'../../input/data/train_images/{idx}'
        extract_frames(os.path.join(input_videos_path,vid_path), train_images_dir)

        driver_rois_dir = f'../../input/data/train_images/{idx}/driver'
        shotgun_rois_dir = f'../../input/data/train_images/{idx}/shotgun'
        
        os.makedirs(driver_rois_dir, exist_ok=True)
        os.makedirs(shotgun_rois_dir, exist_ok=True)

        for filename in os.listdir(train_images_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(train_images_dir, filename)
                print("######## Image Frame Path for classification: ", image_path)
                img = cv2.imread(image_path)
                marked_img, f1, f2, p1, p2 = detect_and_draw_faces(img)
                if p1 and p2:
                    d_roi = marked_img[p1[1]:p1[3], p1[0]:p1[2]]
                    s_roi = marked_img[p2[1]:p2[3], p2[0]:p2[2]]
                    if d_roi.size > 0 and s_roi.size > 0:
                        d_roi_path = os.path.join(driver_rois_dir,f"driver_{filename}")
                        s_roi_path = os.path.join(shotgun_rois_dir, f"shotgun_{filename}")
                        cv2.imwrite(d_roi_path, d_roi)
                        cv2.imwrite(s_roi_path, s_roi)
                    else:
                        print(f"Empty ROI detected for {filename}")
                else:
                    print(f"Invalid coordinates for {filename}")


