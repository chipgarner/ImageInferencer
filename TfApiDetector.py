import os
import time

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util


class TfApiDetector:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)

        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        PATH_TO_MODEL_DIR = '/home/jkg/.keras/datasets/ssd_mobilenet_v2_coco_2018_03_29'
        PATH_TO_LABELS = '/home/jkg/.keras/datasets/mscoco_label_map.pbtxt'
        self.PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                            use_display_name=True)

        matplotlib.use('TkAgg')

        self.detect_fn = None
        self.load_model()

    def load_model(self):
        print('Loading model...', end='')
        start_time = time.time()

        model = tf.saved_model.load(self.PATH_TO_SAVED_MODEL)
        self.detect_fn = model.signatures['serving_default']

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))

    @staticmethod
    def load_image_into_numpy_array(path):
        return np.array(Image.open(path))

    def get_categorie_names(self):
        return None

    def run_inference(self, image_np):
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        start_time = time.time()
        detections = self.detect_fn(input_tensor)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Inference Took {} seconds'.format(elapsed_time))

        formatted_detections = self.format_detections(detections)

        return formatted_detections

    def format_detections(self, detections):
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        return detections
