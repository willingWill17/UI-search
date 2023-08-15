# import thu vien
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.get_extract_model()
    def get_extract_model(self):
        vgg16_model = VGG16(weights="imagenet")
        self.extract_model = Model(inputs=vgg16_model.inputs,
                            outputs=vgg16_model.get_layer("fc1").output)
       

    def image_preprocess(self, img):
        img = img.resize((224, 224))
        img = img.convert("RGB")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.extract_model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature 




