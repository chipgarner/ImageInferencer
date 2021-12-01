import time
import Paths
import numpy as np

from MaskRCNN import model as modellib
from MaskRCNN.config import Config

import logging


class Detector:
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Root directory of the project
        ROOT_DIR = Paths.this_directory()

        # Directory to save logs and trained model TODO fix logging
        MODEL_DIR = ROOT_DIR + "/logs"

        # Path to trained weights file
        # Download this file and place in the root of your
        # project (See README file for details)
        COCO_MODEL_PATH = ROOT_DIR + "/MaskRCNN/mask_rcnn_coco.h5"

        # Directory of images to run detection on
        self.IMAGE_DIR = ROOT_DIR + "/images"

        class InferenceConfig(Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            NAME = "coco"
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + 80

        config = InferenceConfig()
        # config.print_config()

        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        # below is the magic to solve TF's tensor is not an element of this graph issue
        _ = self.model.detect([np.zeros((720, 1280, 3))], verbose=0)

    def detect(self, image):
            start_time = time.time()

            results = self.model.detect([image], verbose=0)

            self.logger.info("--- %s seconds ---" % (time.time() - start_time))

            return results[0]

