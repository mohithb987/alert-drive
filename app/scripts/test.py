import os
from extract_roi_images import extract_frames

test_videos_dir = '../../input/data/videos/test'

for idx, vid_path in enumerate(os.listdir(test_videos_dir)):
    extended_path = os.path.join(test_videos_dir,vid_path)
    output_directory = f'../../input/data/test_images/{idx}'
    os.makedirs(output_directory, exist_ok=True)
    extract_frames(extended_path, output_directory)
