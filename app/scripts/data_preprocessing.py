from PIL import Image
import os

def resize_images(input_directory, output_directory, size=(256, 256)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg')):
            with Image.open(os.path.join(input_directory, filename)) as img:
                img_resized = img.resize(size)
                img_resized.save(os.path.join(output_directory, filename))
            print(f"Resized and saved {filename}")