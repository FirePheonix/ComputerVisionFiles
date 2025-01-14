import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

#below are some prerequisites we have to make before working on mediapipe
#read Hands() documentation before this.
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#we would need A LOT of mathematics to draw lines between two pointers
#so, mediapipe provided us with a function:
mpDraw = mp.solutions.drawing_utils


#for FRAME PER SECOND
pTime = 0
cTime = 0


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #we converted it since hands ONLY uses RGB objects

    results = hands.process(imgRGB)

    #this prints NONE if no hand detected, but prints some coordinates if hand detected
    #print(results.multi_hand_landmarks)
    #so we run it to test wether our hand detection works or not.

    if results.multi_hand_landmarks :
        #handLms -> means hand Landmarks
        for handLms in results.multi_hand_landmarks:

            #now we add functionality for tracking hand landmarks
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                
                #we will find the (x,y) location of the landmarks in order to find out the information on height
                #the location id, lm given back is in decimals
                #so, we multiple the height and width with the decimals in order to find the info
                h, w, c = img.shape
                cx, cy = int(lm.x*w) , int(lm.y*h)
                print(id, cx,cy) # id is printed along with cx,cy in order to tell which exact landmark's position it is

                if id == 4:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
