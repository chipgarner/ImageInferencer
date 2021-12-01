import ImageProvider
import threading
import time
import logging
import cv2
import numpy as np
import requests


class IpCamera(ImageProvider.ImageProvider):
    def __init__(self, ip_address):
        self.logger = logging.getLogger(__name__)

        self.latest_image = None
        self.new_image = threading.Event()

        self.video_delay = 0
        self.consecutive_wait_timeouts = 0

        self.frame_good = True
        self.cv2_capture = None
        self.cam_running = True
        self.keep_running = True

        self._start_ip_camera(ip_address)

    def stop(self):
        self.keep_running = False

    def _start_ip_camera(self, camera_url):
        self.camera_url = camera_url
        time.sleep(0.4)
        self.response_stream = requests.get(camera_url, stream=True, timeout=5.0)

        if self.response_stream.status_code == 200:
            self.cam_thread = threading.Thread(target=self.__camera_thread, name='ipcam')
            self.cam_thread.start()
        elif self.response_stream.status_code == 401:
            self.logger.error("Camera, admin or password wrong.")
        else:
            self.logger.error("Camera. Received unexpected status code {}".format(self.response_stream.status_code))

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

    def __camera_thread(self):
        # About 0.2 seconds per frame, same on both PC and R Pi
        cam_bytes = bytes()

        while self.keep_running:
            # 2048 works, slower with very large or very small chunk size
            try:
                chunk = next(self.response_stream.iter_content(chunk_size=2048))
            except Exception as ex:
                self.logger.error('Exception in camera read loop ' + str(ex))
                self.logger.error('Attempting to recover')

                self.response_stream.close()
                time.sleep(0.3)
                self.response_stream = requests.get(self.camera_url, stream=True, timeout=5.0)
                time.sleep(0.5)

                continue

            cam_bytes += chunk
            a = cam_bytes.find(b'\xff\xd8')
            b = cam_bytes.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = cam_bytes[a:b + 2]
                cam_bytes = cam_bytes[b + 2:]
                try:
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        self.latest_image = image
                        self.__got_goodframe()
                except Exception as ex:
                    # Just keeps trying
                    self.logger.error('Exception in cv2.imdecode ' + str(ex))

        self.latest_image = None
        self.logger.error('Camera thread stopping')

    def __got_goodframe(self):
        if self.latest_image is not None:
            self.new_image.set()

            self.frame_good = True
            self.cam_running = True
        else:
            self.logger.error('got_good_frame but latest_image is None.')




