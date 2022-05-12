#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow import keras
import numpy as np
import cv2
import os

# Resnet18_V1
class Model:

    saved_angle_model = 'angle_Resnet18_v1.h5'
    saved_speed_model = 'speed_Resnet18_v1.h5'

    def __init__(self):
        self.angle_model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_angle_model))
        self.speed_model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_speed_model))

    def preprocess(self, image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)) / 255
        return image

    def predict(self, image):
        image = self.preprocess(image)
        pred_angle = self.angle_model.predict(np.asarray(image))[0]
        pred_speed = self.speed_model.predict(np.asarray(image))[0]
        angle = np.argmax(pred_angle) * 5 + 50
        speed = np.argmax(pred_speed) * 35
        return angle, speed

