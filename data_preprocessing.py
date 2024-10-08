import cv2
import numpy as np
import os

def load_data(img_size=300):
    categories = ['rock', 'paper', 'scissors']
    data = []

    for category in categories:
        folder_path = os.path.join('data', category)
        class_num = categories.index(category) 
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            img_array = np.array(img) / 255.0 
            data.append([img_array, class_num])

    np.random.shuffle(data)

    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)

    return X, y
