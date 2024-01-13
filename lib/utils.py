import cv2
from PIL import Image
import numpy as np
import tkinter as tk

class ImageUtils:

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
    def ToOpenCV(pil_image):
        """
        Converts a PIL image to OpenCV format.
        
        :param pil_image: PIL image
        :return: OpenCV image
        """
        # Convert PIL image to RGB if it's not in that mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        # Convert to OpenCV image
        opencv_image = np.array(pil_image)
        # Convert from RGB to BGR
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        return opencv_image