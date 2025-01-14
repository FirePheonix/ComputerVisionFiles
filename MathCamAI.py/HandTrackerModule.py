import cv2
import mediapipe as mp
import time


class handDetector(): 
   
   
   
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5): #we defined a function here with boiler plate autocomplete parameters, SO THAT WE CAN MANIPULATE those parameters as well
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        #below are some prerequisites we have to make before working on mediapipe
        #read Hands() documentation before this.
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )

        #we would need A LOT of mathematics to draw lines between two pointers
        #so, mediapipe provided us with a function:
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

        #since tipIds are not going to change, so we'll declare them in initialization method
        self.tipIds = [4, 8, 12, 16, 20]
   
   
   
    def findHands(self,img, draw = True):
        #this function need image, and returns the lines over hand.


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        #this prints NONE if no hand detected, but prints some coordinates if hand detected
        #print(results.multi_hand_landmarks)
        #so we run it to test wether our hand detection works or not.
 
        if self.results.multi_hand_landmarks :

            #handLms -> means hand Landmarks
            for handLms in self.results.multi_hand_landmarks:
                            
                # #we will find the (x,y) location of the landmarks in order to find out the information on height
                # #the location id, lm given back is in decimals
                # #so, we multiple the height and width with the decimals in order to find the info
                # h, w, c = img.shape
                # cx, cy = int(lm.x*w) , int(lm.y*h)
                # print(id, cx,cy) # id is printed along with cx,cy in order to tell which exact landmark's position it is

                # if id == 4:
                #    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList
    

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        
        return fingers


            




def main():


    #for FRAME PER SECOND
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        lmList = detector.findPosition(img)


        if len(lmList) != 0:
            print(lmList[4])  # Print thumb tip position

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow("Image", img)

        if not success:
            print("Failed to capture frame")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     

    cap.release()
    cv2.destroyAllWindows()
 



if __name__ == "__main__":
    main()
