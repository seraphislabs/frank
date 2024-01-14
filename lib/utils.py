import cv2
from PIL import Image
import numpy as np
import tkinter as tk

class BboxUtils:
    @staticmethod
    def bbox_is_inside_track(track_bbox, face_bbox):
        track_x1, track_y1, track_x2, track_y2 = track_bbox

        # Unpack the bounding box and ignore the additional values
        (face_x1, face_y1, face_x2, face_y2), _, _ = face_bbox

        # Check if face box is inside track box
        return (face_x1 >= track_x1 and face_y1 >= track_y1 and
                face_x2 <= track_x2 and face_y2 <= track_y2)

class TextUtils:
    @staticmethod
    def Debug(text, override=False):
        """
        Prints a debug message to the console.
        
        :param text: Text to print
        """
        if override is True:
            print(f"{text}", flush=True, end="")
        else:
            print(f"{text}")

class ImageUtils:

    @staticmethod
    def CropBbox(image, bbox):
        left, top, right, bottom = bbox

        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(left, min(right, width))
        bottom = max(top, min(bottom, height))

        cropped_image = image[top:bottom, left:right]
        return cropped_image

    @staticmethod
    def ToPil(opencv_image):
        """
        Converts an OpenCV image to PIL format.
        
        :param opencv_image: OpenCV image
        :return: PIL image
        """
        # Convert from BGR to RGB
        color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL image
        pil_image = Image.fromarray(color_converted)
        return pil_image
    
    @staticmethod
    def ToKinterPhoto(opencv_image):
        """
        Converts an OpenCV image to a tkinter PhotoImage.
        
        :param opencv_image: OpenCV image
        :return: tkinter PhotoImage
        """
        pil_image = ImageUtils.ToPil(opencv_image)
        tk_image = tk.PhotoImage(pil_image)
        return tk_image
    
    @staticmethod
    def ToOpenCV(image):
        """
        Converts a PIL image to OpenCV format.
        
        :param pil_image: PIL image
        :return: OpenCV image
        """
        if isinstance(image, np.ndarray):
        # If the image is already a NumPy array, return it directly
            return image

        # If the image is a PIL image, convert it to OpenCV format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)[..., ::-1]