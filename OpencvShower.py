import cv2
import numpy as np


class OpencvShower:
    def __init__(self, name, full_screen=True):
        if full_screen:
            pass
            # cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.name = name

    def display_images_side_by_side(self, image_right, image_left):
        if image_right is None:
            return
        if image_left is None:
            image_left = image_right

        height, width, channels = image_right.shape
        wide_image = np.zeros((height, width * 2, channels), np.dtype('B'))

        wide_image[:height, :width, :channels] = image_right
        wide_image[:height, width:width*2, :3] = image_left

        show_image = cv2.cvtColor(wide_image, cv2.COLOR_RGB2BGR)

        cv2.imshow(self.name, show_image)
        cv2.waitKey(1)

    def display_image(self, np_image):
        if np_image is not None:
            show_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

            cv2.imshow(self.name, show_image)
            cv2.waitKey(1)
