import PeopleDetection
import threading
import ImageProvider
import time
import LocalInference
import json


class LocalDetection(PeopleDetection.PeopleDetection):
    def __init__(self, image_getter):
        super().__init__(image_getter)
        self.people_caller = None

    def start_recognizing(self):
        recognizer_thread = threading.Thread(target=self.__get_recognize_image, name='recognizer')
        recognizer_thread.start()

    def __get_recognize_image(self):
        local_inference = LocalInference.LocalInference()
        image_provider = ImageProvider.Context(self.cam)

        while True:
            self.logger.info('***************************    START       ****************************')

            start = time.time()

            np_image = image_provider.get_the_next_image()

            start_api = time.time()
            detection_results = local_inference.do_next_image(np_image)

            if detection_results is not None:
                if 'Read timed out' in detection_results or 'Max retries exceeded' in detection_results:
                    self.logger.error('ERROR - timed out.')
                else:
                    self.logger.info("Inference turnaround time: " + str(time.time() - start_api))

                    self.output_time = time.time()
                    self.output_image = self.draw_bbxs(detection_results, np_image, local_inference.get_categorie_names())

                    self.logger.info('Total recognize time: ' + str(time.time() - start))
                    # self.logger.debug(detection_results)

            if detection_results is None:
                self.logger.error('None from get_result')

            time.sleep(3)

    def draw_bbxs(self, image_results, np_image, class_names):

        use_classes = ['BG', 'person', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'teddy bear']

        if type(image_results) is dict or type(image_results) is list:
            results = image_results
        else:
            try:
                results = json.loads(image_results)
            except Exception as ex:
                print('Error: ' + image_results)
                raise(ex)

        categories = []
        for id in results['class_ids']:
            categories.append(class_names[id])

        self.logger.debug('Categories found: ' + str(categories))

        if 'rois' in results:
            bboxes = results['rois']

            for index, bbx in enumerate(bboxes):
                label = categories[index]
                if label in use_classes:
                    color = (255, 0, 0)
                    self.draw_bounding_box(bbx, label, color, np_image)
                    #TODO save the image. Create motion triggeered batches and save surrounding images
        else:
            self.logger.error('*****************************  No Bounding Boxes Found *********************')
        return np_image
