import cv2
import numpy as np
import time
import os
import HandTrackerModule as htm

brushThickness = 10
eraserThickness = 50


folderPath = "Header"
myList = os.listdir(folderPath)
#print(myList)
overlayList=[]

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}') 
    overlayList.append(image)

#print(len(overlayList))

header = overlayList[0]
drawColor = (0,0,255) #BGR

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon = 0.85)
xp, yp = 0,0

#when we draw, we want our drawing to persist and not go away as xp yp and x1 y1 updates
#so we make another one:
imgCanvas = np.zeros((720, 1280, 3), np.uint8) #uint8 means color range till 255


while True:

    #1) import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # flipCode = 0: Flip the image vertically.
    # flipCode > 0: Flip the image horizontally.
    # flipCode < 0: Flip the image both horizontally and vertically.



    #2) find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:
        #print(lmList)

        #tip of index and middle finger.
        x1, y1 = lmList[8][1 : ]
        x2, y2 = lmList[12][1 : ]



        #3) Check which fingers are up 

        fingers = detector.fingersUp()
        #print(fingers)

        #4) if selection mode- two fingers are up, then we're selected and not drawing
        if fingers[1] and fingers[2]:
            
            #print("selection mode")

            #wheneverhand there is a selection, change the xp and yp = 0,0 which further changes to x1 and y1
            xp, yp = 0,0 



            #checking the clicks:
            if x1 < 108  and 0< y1 < 500:
                if  0 < y1 < 100 :
                    header = overlayList[0]
                    drawColor = (0,0,255)
                elif 100 < y1 < 208 :
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 208 < y1 < 308 :
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 308 < y1 < 400:
                    header = overlayList[3]
                    drawColor = (128, 128, 128) #gray
                elif 400 < y1 < 500:
                    header = overlayList[4]
                    drawColor = (0,0,0)
                    
            cv2.rectangle(img, (x1, y1 - 25), (x2 , y2 + 25), drawColor, cv2.FILLED)

            
            



        #5) if drawing mode - index finger up. 
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1) , 15, drawColor, cv2.FILLED)
            #print("drawing mode")

            #we just draw a line by taking 2 very close ponits and draw many of them like that.
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, eraserThickness)
            else: 
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brushThickness)
            


            #updating the previous point as the new point
            xp, yp = x1, y1

            # Clear Canvas when all fingers are up
        if all (x >= 1 for x in fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)


    # Create a mask where the canvas has non-black pixels
    canvasMask = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, canvasMask = cv2.threshold(canvasMask, 10, 255, cv2.THRESH_BINARY)

    # Invert the mask
    canvasMaskInv = cv2.bitwise_not(canvasMask)

    # Use the mask to keep the original image in areas where there's no drawing
    imgBackground = cv2.bitwise_and(img, img, mask=canvasMaskInv)

    # Use the canvas mask to extract the drawing
    imgForeground = cv2.bitwise_and(imgCanvas, imgCanvas, mask=canvasMask)


    # Combine the original frame and canvas using addWeighted
    img = cv2.add(imgBackground, imgForeground)




    #setting the header
    img[0:496 , 0:108] = header


    cv2.imshow("Image", img)
    #cv2.imshow("ImageCanvas", imgCanvas)
    cv2.waitKey(1)