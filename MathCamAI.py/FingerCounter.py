import cv2
import time
import os
import HandTrackerModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam) #defined width of the cam
cap.set(4, hCam) #defined height of the cam

folderPath = "fingerImages"
myList = os.listdir(folderPath)
print(myList)

#we created an array of images. named it as OVERLAY since we have to OVERLAY it on top of images.
overlayList = []


for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}') to check if path works.
    overlayList.append(image) #inserted the image paths to overlayList array.

print(len(overlayList))


pTime=0


detector = htm.handDetector(detectionCon = 0.75)
tipIds = [4, 8, 12, 16, 20]



while True:
    succes, img = cap.read()

    image = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    #print(lmList)

    #now, open documentation of the mediapipe webpage to see what hand positions are where
    # example: if 8(tip of index finger) is below 6(bottom of index finger), then your finger is down. :)
    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

            # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
        else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w] = overlayList[totalFingers]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)
            


    #image itself is a matrix, so we need to define a matrix in order to get the image on the screen
    #we will give the range of the height, then the range of the width
    
    # Resize the overlay image to fit the desired region (300x125)
    #resizedOverlay = cv2.resize(overlayList[1], (125, 300))  # width=125, height=300
    #img[0:300, 0:125] = resizedOverlay  # Assign the resized overlay to the region



    #but we can have different sizes of images, so we provide a RANGE.
    #h ,w, c = overlayList[0].shape
    #img[0:h , 0:w] = overlayList[0]

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'Fps: {fps}', (400,70), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0) , 3)
    
    cv2.imshow("image", img)
    cv2.waitKey(1) #gives one milisecond delay to frame reading
