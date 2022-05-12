#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow import keras
import numpy as np
import cv2
import os

# angle: Resnet152_v2, speed: Resnet18_v2
class Model:

    saved_angle_model = 'angle_Resnet152_v2.h5'
    saved_speed_model = 'speed_Resnet18_v2'

    def __init__(self):
        self.angle_model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_angle_model))
        self.speed_model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_speed_model))

    def angle_preprocess(self, image):
        angle_image = cv2.cvtColor(angle_image, cv2.COLOR_BGR2RGB)
        angle_image = cv2.resize(angle_image, (224, 224)) / 255
        return angle_image
    
    def speed_preprocess(self, image):
        speed_image = cv2.cvtColor(speed_image, cv2.COLOR_BGR2RGB)
        speed_image = cv2.GaussianBlur(speed_image, (3,3), 0)
        speed_image = cv2.resize(speed_image, (224, 224)) / 255
        return speed_image

    def predict(self, image):
        angle_image = self.angle_preprocess(image)
        speed_image = self.speed_preprocess(image)
        pred_angle = self.angle_model.predict(np.asarray(angle_image))[0]
        pred_speed = self.speed_model.predict(np.asarray(speed_image))[0]
        angle = np.argmax(pred_angle) * 5 + 50
        speed = np.argmax(pred_speed) * 35
        return angle, speed

