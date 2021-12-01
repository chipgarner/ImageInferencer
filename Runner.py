import ImageProvider
import IpCamera
import LocalDetection
import OpencvShower


class Runner:
    def __init__(self, image_getter, image_shower):
        self.cam = image_getter
        self.shower = image_shower

    def run(self, sentry_id, two_screens=False, branch='dev'):
        image_provider = ImageProvider.Context(self.cam)
        shower = self.shower
        recognizer = LocalDetection.LocalDetection(self.cam)
        recognizer.start_recognizing()

        while True:
            if two_screens:
                next_image = image_provider.get_the_next_image()
                shower.display_images_side_by_side(next_image, recognizer.output_image)
            else:
                shower.display_image(recognizer.output_image)


if __name__ == '__main__':
    runner = Runner(IpCamera.IpCamera('http://admin:@192.168.1.53/video/mjpg.cgi'), OpencvShower.OpencvShower('Video'))
    # runner = Runner(UsbCamera.UsbCamera(), OpencvShower.OpencvShower('Video', full_screen=False))
    # runner = Runner(ImageRepeater.ImageRepeater(), OpencvShower.OpencvShower('Video', full_screen=False))

    # Produciton: 'JW2ZKG-B'  #'5FBUD6-B' # Develop:'S-J3FKD5'
    # prod and dev 'AVWMT8-B'
    # staging 'YX3HRM8E-P'
    #'3JP3AKEX-P' #NBLTTGRA-P'  #KQP5W35N-P' #  #'TESTATEST' #''AVWMT8-B'
    # #FrontDoor" 'S-J3FKD5'  #'5FBUD6-B' #'S-J3FKD5'  # '5FBUD6-B'

    # sentry_id = 'AVWMT8-B' # prod and dev
    sentry_id = '5FBUD6-B' # V2
    # sentry_id = 'YX3HRM8E-P' #  'staging

    runner.run(sentry_id, two_screens=True, branch='dev')
