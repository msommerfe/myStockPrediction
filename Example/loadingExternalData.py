import numpy as np
import os
import cv2
import random

DATADIR = "D:/dev/Datasets/PetImages"
CATEGORIES = ["Dog","Cat"]
IMG_SIZE = 50
training_data = []
X = []
Y = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        Class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array =cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,Class_num])
            except Exception as e:
                pass

    random.shuffle(training_data)



    for feature, label in training_data:
        X.append(feature)
        Y.append(label)


create_training_data()

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,1)



np.save('features.npy',X)
np.save('labels.npy',Y)
X=np.load('features.npy')
Y=np.load('labels.npy')

