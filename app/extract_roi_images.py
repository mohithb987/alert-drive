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

        if frame_count % fps == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    video_capture.release()
    print(f"Extracted {saved_frame_count} frames from {video_path} to {output_dir}")


# video_path = '../input/data/videos/train/7.mp4'
# output_dir = '../input/data/images/train'
# extract_frames(video_path, output_dir)


if __name__ == "__main__":
    import os
    import sys
    import cv2
    from detect_and_draw import detect_and_draw_faces

    driver_rois_dir = '../input/data/images/driver'
    shotgun_rois_dir = '../input/data/images/shotgun'
    input_dir = '../input/data/images/train'
    os.makedirs(driver_rois_dir, exist_ok=True)
    os.makedirs(shotgun_rois_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_dir, filename)
            print("######## Path: ", image_path)
            img = cv2.imread(image_path)
            marked_img, p1, p2 = detect_and_draw_faces(img)
            if p1 and p2:
                d_roi = marked_img[p1[1]:p1[3], p1[0]:p1[2]]
                s_roi = marked_img[p2[1]:p2[3], p2[0]:p2[2]]
                if d_roi.size > 0 and s_roi.size > 0:
                    d_roi_path = os.path.join(driver_rois_dir, filename)
                    s_roi_path = os.path.join(shotgun_rois_dir, filename)
                    cv2.imwrite(d_roi_path, d_roi)
                    cv2.imwrite(s_roi_path, s_roi)
                else:
                    print(f"Empty ROI detected for {filename}")
            else:
                print(f"Invalid coordinates for {filename}")


