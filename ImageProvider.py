from abc import ABC, abstractmethod


class ImageProvider(ABC):
    @abstractmethod
    def get_next_image(self):
        # Return an image as an RGB np array
        pass

    @abstractmethod
    def stop(self):
        pass


class Context():
    """
    Strategy desing pattern
    The Context defines the interface of interest to clients.
    """

    def __init__(self, image_provider: ImageProvider) -> None:
        self._image_provider = image_provider

    @property
    def get_image_provider(self) -> ImageProvider:
        return self._image_provider

    def set_image_provider(self, image_provider: ImageProvider) -> None:
        """
        Allows replacing a Strategy object at runtime.
        """

        self._image_provider = image_provider

    def get_the_next_image(self):
        return self._image_provider.get_next_image()

    def stop_camera(self):
        return self._image_provider.stop()
