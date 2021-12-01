import MaskRCNN.detector as detector


class LocalInference:
    def __init__(self):
        self.detector = detector.Detector()
        self.detect = self.detector.detect

    def do_next_image(self, np_image):
        results = self.detect(np_image)
        return results

    def get_categorie_names(self):
        return self.detector.class_names