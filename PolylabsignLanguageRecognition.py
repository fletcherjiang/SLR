import os
import cv2
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import numpy as np, random, time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

cap = cv2.VideoCapture(1)
WindName = "Sign Language Recognition @COMP4423 JIANG Yiyang"
cv2.namedWindow(WindName)
cv2.resizeWindow(WindName, 1024, 768)

landmarks_on=True # draw landmarks or not
extract_mode_on=True # put game in extract or testing mode

polysmart_svr_url = 'http://158.132.255.32:8088/'
polysmart_facerecg_svr = polysmart_svr_url+'handdetect/'


def detect(pic):
    _, im_buf = cv2.imencode(".jpg", pic)
    byte_im = im_buf.tobytes()

    data = MultipartEncoder(fields={'file': ('img.jpg', byte_im)})
    response = requests.post(polysmart_facerecg_svr, data=data, headers={
        'Content-Type': data.content_type})
    retJson = response.json()
    return retJson['results'] if retJson['code'] >= 0 else []

connections = [[4, 3, 2, 1, 0],# thumb
               [8, 7, 6, 5],# index
               [12, 11, 10, 9],# middle
               [16, 15, 14, 13],# ring
               [20, 19, 18, 17, 0], #pinky
               [3, 5, 9, 13, 17]# palm
               ]

def draw_landmarks(image, landmarks):
    h, w, c = image.shape
    # print([h, w, c])
    id2cords = {}
    for lm in landmarks:
        idx, ftx, fty = lm['idx'], int(lm['x']*w), int(lm['y']*h)
        id2cords[idx] = [ftx, fty]
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

#28x28 image feature extraction, but at the end abandon this fang
def handextrac_feature(hand):
    feat=[]
    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY) 
    #hand = cv2.GaussianBlur(hand,(5,5),0)
    #hand = cv2.bilateralFilter(hand,9,75,75)
    hand = cv2.medianBlur(hand, 5)
    hand = cv2.resize(hand,(28,28))
    feat = hand.reshape(1,-1)
    return feat[0]/255



feat_x,feat_y=np.load('feat_x.npy'),np.load('feat_y.npy')
sign = ['0','a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

######################### run cross-validation #########################
from sklearn import svm
from sklearn.model_selection import cross_val_score
model = svm.SVC(kernel='rbf')
scores = cross_val_score(model, feat_x, feat_y, cv=10)
total=0
for i in scores:
    total+=i
avg = total/len(scores)

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

label2str={1:'A',2:'B',3:'C',4:"D",5:"E",6:"F",7:"G",8:"H",9:"I",10:"K",11:"L",12:"M",13:"N",14:"O",15:"P",
           16:"Q",17:"R",18:"S",19:"T",20:"U",21:"V",22:"W",23:"X",24:"Y"}

######################### Fit the model #########################
model.fit(feat_x,feat_y)
y_pred = model.predict(feat_x)
from sklearn import metrics
accur = metrics.accuracy_score(y_true=feat_y, y_pred=y_pred)
print("accuracy:", metrics.accuracy_score(y_true=feat_y, y_pred=y_pred), "\n")

######################### Testing Mode or Extract Mode #########################
status_texts=['Testing Mode', 'Extract Mode']
status_durations=[0,4,1,1,2,4] # how long each status lasts (in seconds)
status_check_points=[sum(status_durations[:i+1]) for i in range(len(status_durations))]
status_id = 0
refesh = 0
signstate = 0
savestate = 0

while True:
    success, image = cap.read()
    if not success:
        print("Open cam failed ....")
        continue

    #Together with the feature extraction function, can test load new models at the same time
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

    image = cv2.flip(image, 1)
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
    id2cords = {}
    results = detect(imageRGB)

        
    if len(results) == 0:
        print("Nothing detected ...")
        image=cv2.putText(image,"Nothing detected",(1620,300),cv2.FONT_HERSHEY_TRIPLEX,0.9,(65, 79, 255),2)

    #Print the recognized letters
    else:
        image, id2cords = draw_landmarks(image, results[0]['landmarks'])
        feat=np.array([extrac_feature(id2cords)])
        label=model.predict(np.array(feat))[0]
            
        if status_id==0:
            image=cv2.putText(image,label2str[label],(1620,330),cv2.FONT_HERSHEY_TRIPLEX,3,(65, 79, 255),3)
 
     
    cv2.imshow(WindName, image)
    #print('Show image done!') 

    key=cv2.waitKey(1) & 0xFF

    if key == ord('3'):
        refesh = 1


    if key == ord('0') or key==27:
        break
    
    #Feature extraction function
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


            results = detect(imageRGB)
            id2cords = {}
            if len(results) == 0:
                image=cv2.putText(image,"Nothing detected",(1620,250),cv2.FONT_HERSHEY_TRIPLEX,0.9,(65, 79, 255),2)
            
            else:
                image, id2cords = draw_landmarks(image, results[0]['landmarks'])
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
                    print("This is the Sign", sign[i])
                    signcharacter = sign[i].upper()

            if key == ord('1') and len(feat_x)>0:
                savestate = 1
                np.save('feat_x.npy',np.array(feat_x))
                np.save('feat_y.npy',np.array(feat_y))
                print('feature saved ...')
          
    if key == ord('2'):
        landmarks_on=not landmarks_on
        
     
# release the cap object
cap.release()
# destroy all the windows
cv2.destroyAllWindows()