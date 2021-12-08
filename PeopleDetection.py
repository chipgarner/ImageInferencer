import cv2
from sentrycommon import Logger
import logging


class PeopleDetection:
    def __init__(self, image_getter):

        LogSetup = Logger.Logger_Setup()
        self.logger = logging.getLogger(__name__)
        LogSetup.set_level(self.logger)

        self.cam = image_getter

        self.output_image = None
        self.output_time = 0

    def draw_bounding_box(self, bounding_box, label, color, np_image):

        top = bounding_box[0]
        left = bounding_box[1]
        bottom = bounding_box[2]
        right = bounding_box[3]

        if not isinstance(top, int):
            top, left, bottom, right = self.convert_bbx_to_pixels(bounding_box, np_image.shape)

        cv2.rectangle(np_image, (left, top), (right, bottom), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(np_image, label, (left, top - 5), font, 1, color, 2, cv2.LINE_AA)

    def convert_bbx_to_pixels(self, bbx, image_shape):
        height, width, _ = image_shape
        top = int(height * bbx[0])
        left = int(width * bbx[1])
        bottom = int(height * bbx[2])
        right = int(width * bbx[3])
        return top, left, bottom, right


if __name__ == '__main__':
    import ImageRepeater

    cam = ImageRepeater.ImageRepeater()
    recognizer = PeopleDetection(cam, None)
    recognizer.start_recognizing()
