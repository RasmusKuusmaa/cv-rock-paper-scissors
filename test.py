import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
from tensorflow.keras.models import load_model
import random

model = load_model('rps2_model.keras')

labels = ['rock', 'paper', 'scissors']


cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300


frame_skip = 24
frame_count = 0  

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break


    hands, img = detector.findHands(img)

    if frame_count % frame_skip == 0 and hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.size > 0:
            imgCrop = cv.resize(imgCrop, (imgSize, imgSize))
            imgWhite[0:imgCrop.shape[0], 0:imgCrop.shape[1]] = imgCrop

            imgArray = np.array(imgCrop) / 255.0  
            imgArray = np.expand_dims(imgArray, axis=0)  


            prediction = model.predict(imgArray)
            class_index = np.argmax(prediction, axis=1)[0]
            label = labels[class_index]

         
            cv.putText(img, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(label)
            computer_choiche = random.choice(labels)
            print('computers choiche:', computer_choiche)
            if computer_choiche == 'rock':
                if label == 'paper':
                    print('player won')
                elif label == 'rock':
                    print('ties')
                else:
                    print('computer won')
            elif computer_choiche == 'paper':
                if label == 'rock':
                    print('computer won')
                elif label == 'paper':
                    print('tie')
                else:
                    print('player won')
            elif computer_choiche == 'scissors':
                if label =='paper':
                    print('computer own')
                elif label == 'rock':
                    print('player won')
                else:
                    print('tie')
            
            cv.imshow("Cropped Hand", imgWhite)



    cv.imshow("Video Feed", img)

    frame_count += 1


    key = cv.waitKey(1)
    if key == ord('q'):
        break  

cap.release()
cv.destroyAllWindows()
