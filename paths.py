import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_VIDEOS_PATH = os.path.join(ROOT_DIR, 'input', 'data', 'videos', 'train')
INTERMEDIATE_ROI_PATH = os.path.join(ROOT_DIR, 'input', 'data', 'intermediate_results')
SAVED_MODELS_DIR_PATH = os.path.join(ROOT_DIR, 'model', 'trained_model')
INTERMEDIATE_CLASSIFICATION_PATH = os.path.join(ROOT_DIR, 'output', 'faces_classification_2')
STEERING_WHEEL_AUGMENTED_IMAGES_PATH = os.path.join(ROOT_DIR, 'input', 'data', 'steering_wheel_2', 'augmented_images_2')