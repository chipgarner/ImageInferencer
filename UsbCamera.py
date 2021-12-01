import ImageProvider
import threading
import time
import logging
import cv2


class UsbCamera(ImageProvider.ImageProvider):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.latest_image = None
        self.new_image = threading.Event()

        self.video_delay = 0
        self.consecutive_wait_timeouts = 0

        self.frame_good = True
        self.cv2_capture = None
        self.cam_running = True
        self.keep_running = True

        self._start_usb_camera(0)

    def get_next_image(self) -> bytes:

        if not self.new_image.wait(2.0):  # 2.0 is 2 second wait timeout if no frames come in
            self.consecutive_wait_timeouts += 1
            if self.consecutive_wait_timeouts > 9:
                self.logger.warning(str(self.consecutive_wait_timeouts) + ' consecutive wait timeouts.')
                self.consecutive_wait_timeouts = 0
            self.frame_good = False

        else:
            self.consecutive_wait_timeouts = 0
        self.new_image.clear()

        np_image = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)

        return np_image

    def stop(self):
        self.keep_running = False

    def _start_usb_camera(self, usb_camera_number):  # usb_camera_number is usually zero
        self._start_camera(usb_camera_number)
        cam_thread = threading.Thread(target=self.__usb_cam_thread, name='usbcam')
        cam_thread.start()
        time.sleep(0.02)  # Take a short break so the camera can get going

    def _start_camera(self, path_or_camera_number):
        self.cv2_capture = cv2.VideoCapture()
        self.cv2_capture.open(path_or_camera_number)

    def __usb_cam_thread(self):

        while self.keep_running:
            time.sleep(self.video_delay)
            self.frame_good, self.latest_image = self.cv2_capture.read()
            if self.frame_good:
                self.__got_goodframe()
            else:
                self.latest_image = None

                if self.cam_running:
                    self.cam_running = False

                time.sleep(2)
                self.cv2_capture.close(0)

        self.logger.error('USB camera thread stopping')

    def __got_goodframe(self):
        if self.latest_image is not None:
            self.new_image.set()

            self.frame_good = True
            self.cam_running = True
        else:
            self.logger.error('got_good_frame but latest_image is None.')


if __name__ == '__main__':
    cam = UsbCamera()

    next_image = cam.get_next_image()

    print(next_image)
