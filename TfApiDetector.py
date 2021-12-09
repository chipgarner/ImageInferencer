import logging
import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util


class TfApiDetector:
    def __init__(self):
        logging.disable(logging.WARNING)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1)
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        PATH_TO_MODEL_DIR = '/home/jkg/.keras/datasets/centernet_hg104_512x512_kpts_coco17_tpu-32'
        self.PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

        self.class_names = self.get_categorie_names()

        self.detect_fn = None
        self.load_model()
        self.model = None

    def load_model(self):
        # print('Loading model...', end='')
        start_time = time.time()

        self.model = tf.saved_model.load(self.PATH_TO_SAVED_MODEL)
        self.detect_fn = self.model.signatures['serving_default']

        end_time = time.time()
        elapsed_time = end_time - start_time
        # print('Done! Took {} seconds'.format(elapsed_time))

    @staticmethod
    def load_image_into_numpy_array(path):
        return np.array(Image.open(path))

    def get_categorie_names(self):
        PATH_TO_LABELS = '/home/jkg/.keras/datasets/mscoco_label_map.pbtxt'
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)
        class_names = {}
        for num in category_index:
            class_names.update({category_index[num]['id']: category_index[num]['name']})

        return class_names

    def get_class_names(self):
        return self.class_names

    def run_inference(self, image_np):
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.detect_fn(input_tensor)

        formatted_detections = self.format_detections(detections)

        return formatted_detections

    def format_detections(self, detections):
        num_detections = 0
        formatted_detections = {'detection_scores':[], 'detection_classes': [],
                                'detection_boxes': [], 'num_detections': 0}
        scores = detections['detection_scores'][0].numpy()
        for index, score in enumerate(scores):
            if score > 0.5:
                num_detections+= 1
                formatted_detections['detection_scores'].append(score)

                detected_class = detections['detection_classes'][0].numpy()[index]
                formatted_detections['detection_classes'].append(detected_class.astype(np.int64))

                detection_boxes = detections['detection_boxes'][0].numpy()[index]
                formatted_detections['detection_boxes'].append(detection_boxes)

        formatted_detections['num_detections'] = num_detections

        return formatted_detections
