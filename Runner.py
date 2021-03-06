import time

import ImageProvider
import IpCamera
import LocalDetection
import OpencvShower


class Runner:
    def __init__(self, image_getter, image_shower):
        self.cam = image_getter
        self.shower = image_shower
        # self.shower.go()

    def run(self, two_screens=False):
        image_provider = ImageProvider.Context(self.cam)
        shower = self.shower
        recognizer = LocalDetection.LocalDetection(self.cam)
        recognizer.start_recognizing()

        while True:
            if two_screens:
                next_image = image_provider.get_the_next_image()
                shower.display_images_side_by_side(next_image, recognizer.output_image)
            else:
                time.sleep(1)
                shower.display_image(recognizer.output_image)


if __name__ == '__main__':
    runner = Runner(IpCamera.IpCamera('http://admin:@192.168.1.53/video/mjpg.cgi'),
                    OpencvShower.OpencvShower('Video', full_screen=False))
    # runner = Runner(UsbCamera.UsbCamera(), OpencvShower.OpencvShower('Video', full_screen=False))
    # runner = Runner(ImageRepeater.ImageRepeater(), OpencvShower.OpencvShower('Video', full_screen=False))

    runner.run(two_screens=True)
