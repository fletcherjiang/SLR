import cv2
import mediapipe as mp
import numpy as np, random, time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
WindName = "Sign Language Recognition @COMP4423 JIANG Yiyang"
cv2.namedWindow(WindName)
cv2.resizeWindow(WindName, 1080, 720)


landmarks_on=True # draw landmarks or not

connections = [[4, 3, 2, 1, 0],# thumb
               [8, 7, 6, 5],# index
               [12, 11, 10, 9],# middle
               [16, 15, 14, 13],# ring
               [20, 19, 18, 17, 0], #pinky
               [3, 5, 9, 13, 17]# palm
               ]


def draw_landmarks(image, landmarks):
    h, w,c = image.shape
    id2cords = {}
    for lm, landmrk in enumerate(landmarks):
        idx, ftx, fty  = lm ,landmrk.x * w, landmrk.y*h
        id2cords[idx] = [int(ftx), int(fty)]

    if not landmarks_on: return image,id2cords

    for line in connections:
        pts = [[id2cords[idx][0], id2cords[idx][1]] for idx in line]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        image = cv2.polylines(image, [pts], False, (0, 0, 250), 4)

    for idx in id2cords:
        image = cv2.circle(image, (id2cords[idx][0], id2cords[idx][1]), 10, (0, 0, 250), 5)

    for idx in range(21):
        image = cv2.circle(image, (id2cords[idx][0], id2cords[idx][1]), 15, (0, 250, 0), 3)
        idx=idx+4
        
    #Draw the square hand frame
    X_max = id2cords[0][0]
    X_min = id2cords[0][0]
    Y_max = id2cords[0][1]
    Y_min = id2cords[0][1]

    #Find the smallest and largest x, and the smallest and largest y
    for idx in id2cords:
        if X_max < id2cords[idx][0]:
            X_max = id2cords[idx][0]
    
    for idx in id2cords:
        if X_min > id2cords[idx][0]:
            X_min = id2cords[idx][0]
    
    for idx in id2cords:
        if Y_max < id2cords[idx][1]:
            Y_max = id2cords[idx][1]
    
    for idx in id2cords:
        if Y_min > id2cords[idx][1]:
            Y_min = id2cords[idx][1]


    NewX = int((X_max+X_min)/2)
    NewY = int((Y_max+Y_min)/2)

    #Take the maximum and minimum points to plot the central extreme points.
    if X_max-X_min > Y_max-Y_min:
        #Calculate the length of the sides of the square
        a = X_max-X_min
        #Performs a slight position shift and is able to cover the entire hand
        image = cv2.rectangle(image, (int(NewX - a*0.6),int(NewY + a*0.6)),
                              (int(NewX+ a*0.6),int(NewY- a*0.6)),(0,0,250),5)
        
    else:
        #Calculate the length of the sides of the square
        a = Y_max-Y_min
        #Performs a slight position shift and is able to cover the entire hand
        image = cv2.rectangle(image, (int(NewX - a*0.6),int(NewY + a*0.6)),
                              (int(NewX+ a*0.6),int(NewY- a*0.6)),(0,0,250),5)

    return image, id2cords

def extrac_feature(id2cords):
    feat=[]
    for id in range(21):
        a=np.array(id2cords[id])
        for tag in range(id+1,21):
            b=np.array(id2cords[tag])
            dist=np.linalg.norm(a-b)/800 
            # normalize the distane in the range of [0,1] by assuming the 800 is the maximum dist possible
            feat.append(dist)
    return feat
  
feat_x,feat_y=np.load('feat_x.npy'),np.load('feat_y.npy')
sign = ['0','a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

model = svm.SVC(kernel='rbf')
scores = cross_val_score(model, feat_x, feat_y, cv=10)
total=0
for i in scores:
    total+=i
avg = total/len(scores)

label2str={1:'A',2:'B',3:'C',4:"D",5:"E",6:"F",7:"G",8:"H",9:"I",10:"K",11:"L",12:"M",13:"N",14:"O",15:"P",
           16:"Q",17:"R",18:"S",19:"T",20:"U",21:"V",22:"W",23:"X",24:"Y"}

model.fit(feat_x,feat_y)
y_pred = model.predict(feat_x)
accur = metrics.accuracy_score(y_true=feat_y, y_pred=y_pred)
print("accuracy:", metrics.accuracy_score(y_true=feat_y, y_pred=y_pred), "\n")

status_texts=['Testing Mode', 'Extract Mode']
status_id = 0
refesh = 0
signstate = 0
savestate = 0

cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    max_num_hands = 2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue


    if refesh == 1:
        feat_x,feat_y=np.load('feat_x.npy'),np.load('feat_y.npy')
        model.fit(feat_x,feat_y)
        y_pred = model.predict(feat_x)
        scores = cross_val_score(model, feat_x, feat_y, cv=10)
        total=0
        for i in scores:
            total+=i
        avg = total/len(scores)
        accur = metrics.accuracy_score(y_true=feat_y, y_pred=y_pred)
        refesh = 0

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image_height, image_width, _ = image.shape


    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay=image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    secondeline = "Mode Type: " + status_texts[status_id]
    thirdline = 'Cross Validation: {:.2%}'.format(avg)
    thirdline2 = 'Accuracy: {:.2%}'.format(accur)
    fourthline = 'Recognized Letters: '
    overlay=cv2.rectangle(overlay, (1280,10),(1900,360),color=(1, 0, 11),thickness=-1)
    image=cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    #Print UI
    image=cv2.putText(image,'[0] Exit; [1] Extract Mode;',(1300,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2) 
    image=cv2.putText(image,'[2] Landmark ON/OFF [3] Refesh',(1300,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2)   
    image=cv2.putText(image,secondeline,(1300,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2)
    image=cv2.putText(image,thirdline,(1300,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2)
    image=cv2.putText(image,fourthline,(1300,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2)
    image=cv2.putText(image,thirdline2,(1300,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2)
    
    if results.multi_hand_landmarks:
      
      for hand_landmarks in results.multi_hand_landmarks:
        # Here is How to Get All the Coordinates
        image, id2cords = draw_landmarks(image, hand_landmarks.landmark)
        if landmarks_on:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        feat=np.array([extrac_feature(id2cords)])
        label=model.predict(np.array(feat))[0]

        if status_id==0:
            image=cv2.putText(image,label2str[label],(1620,330),cv2.FONT_HERSHEY_TRIPLEX,3,(65, 79, 255),3)
          
    else:
        image=cv2.putText(image,"Nothing detected",(1620,300),cv2.FONT_HERSHEY_TRIPLEX,0.9,(65, 79, 255),2)
        
        
        
    # Flip the image horizontally for a selfie-view display.


    cv2.imshow(WindName, image)

    key=cv2.waitKey(1) & 0xFF
    
    if key == ord('3'):
        refesh = 1

    if key == ord('0') or key==27:
        break

    if key == ord('2'):
        landmarks_on=not landmarks_on

    if key == ord('1'):
        feat_x,feat_y=np.load('feat_x.npy').tolist(),np.load('feat_y.npy').tolist()
        status_id == 1
        fifthline = "Number of feature is: " + str(len(feat_x))
        while True: 
            success, image = cap.read()
            if not success:
                continue
            image = cv2.flip(image, 1)
            overlay=image.copy()
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image=cv2.rectangle(overlay, (1280,10),(1900,370),color=(1, 0, 11),thickness=-1)
            image=cv2.putText(image,'[0] Testing Mode; [1] Save data',(1300,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2) 
            image=cv2.putText(image,"Mode Type: Exract Mode",(1300,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2)
            image=cv2.putText(image,'Tap in the corresponding letter',(1300,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2)   
            image=cv2.putText(image,'(exclude J)while show sign language',(1300,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2)
            image=cv2.putText(image,fourthline,(1300,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2)
            image=cv2.putText(image,fifthline,(1300,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 252, 0),2)


            image.flags.writeable = False
            results = hands.process(image)
            image_height, image_width, _ = image.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Here is How to Get All the Coordinates
                    image, id2cords = draw_landmarks(image, hand_landmarks.landmark)
                    if landmarks_on:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                    
                feat=np.array([extrac_feature(id2cords)])
                label=model.predict(np.array(feat))[0]
                image=cv2.putText(image,label2str[label],(1620,260),cv2.FONT_HERSHEY_TRIPLEX,2,(65, 79, 255),3)

                if signstate == 1:
                    signtext = "This is the Sign " + signcharacter
                    image=cv2.putText(image,signtext,(1300,350),cv2.FONT_HERSHEY_TRIPLEX,1,(65, 79, 255),2)
                    image=cv2.putText(image,signtext,(1300,350),cv2.FONT_HERSHEY_TRIPLEX,1,(65, 79, 255),2)
                    signstate = 0
                if savestate == 1:
                    savetext = "Save Successfully!"
                    image=cv2.putText(image,savetext,(1300,350),cv2.FONT_HERSHEY_TRIPLEX,1,(65, 79, 255),2)
                    image=cv2.putText(image,savetext,(1300,350),cv2.FONT_HERSHEY_TRIPLEX,1,(65, 79, 255),2)
                    savestate = 0
                    
            else:
                image=cv2.putText(image,"Nothing detected",(1620,250),cv2.FONT_HERSHEY_TRIPLEX,0.9,(65, 79, 255),2)


            cv2.imshow(WindName, image)

            key=cv2.waitKey(1) & 0xFF
        
            if key == ord('0'):
                status_id=0
                break

            for i in range(len(sign)):
                if key == ord(sign[i]) and not id2cords =={}:
                    signstate = 1
                    feat=extrac_feature(id2cords)
                    feat_x.append(feat)
                    feat_y.append(i)
                    signcharacter = sign[i].upper()

            if key == ord('1') and len(feat_x)>0:
                savestate = 1
                np.save('feat_x.npy',np.array(feat_x))
                np.save('feat_y.npy',np.array(feat_y))

cap.release()
cv2.destroyAllWindows()


