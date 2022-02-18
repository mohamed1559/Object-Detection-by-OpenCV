import cv2
import numpy as np 
thresh = 0.5
nmsThresh = 0.4

# inserting the names from the (file.names) file 
classNames = []
classFile ='file.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')   # each row represent a name of an object 

# for configration & weight 
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'


net = cv2.dnn_DetectionModel(weightPath,configPath) # for detect the model 
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5, 127.5))
net.setInputSwapRB(True)


#=============> for images 

#img = cv2.imread('sheep.jpg')   #read image 

#classIds , confs , bbox = net.detect(img,confThreshold=thresh)                    # for detecting the threshold (if i 50% at least it will be good enough)
#elements = cv2.dnn.NMSBoxes(bbox,confs,thresh,nmsThresh)

#for i in elements:
#     data_matrix = bbox[i]
#     x, y, w, h = data_matrix[0], data_matrix[1], data_matrix[2], data_matrix[3]
#     cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)  # for the rectangle which detect the objects 
#     cv2.putText(img, classNames[classIds[i] - 1], (data_matrix[0], data_matrix[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6,
#                 (0,255,0), 2)                                                   # for the name which we get from (file.names)
#     cv2.putText(img, str(round(confs[i] * 100, 2)) + '%', (data_matrix[0], data_matrix[1] + 50), cv2.FONT_HERSHEY_COMPLEX, 0.6,
#                 (0,255,0), 2)

#cv2.imshow('Output',img)
#cv2.waitKey(0)




# ========>for camera video

cap = cv2.VideoCapture(0)      # for video capture from my PC 

while True:
     success,img = cap.read()
     img = cv2.flip(img, 1)
     classIds , confs , bbox = net.detect(img,confThreshold=thresh)   # for detecting the threshold (if i 50% at least it will be good enough)
     bbox = list(bbox)
     confs = list(np.array(confs).reshape(1,-1)[0])
     confs = list(map(float,confs))

     elements = cv2.dnn.NMSBoxes(bbox,confs,thresh,nmsThresh)

     for i in elements:
         data_matrix = bbox[i]
         x,y,w,h = data_matrix[0],data_matrix[1],data_matrix[2],data_matrix[3]
         cv2.rectangle(img, (x,y),(x+w,y+h), color=(0, 255, 0), thickness=2)  # for the rectangle which detect the objects 
         cv2.putText(img, classNames[classIds[i]-1], (data_matrix[0], data_matrix[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                     (0, 255, 0), 2)                                           # for the name which we get from (file.names)
         cv2.putText(img, str(round(confs[i] * 100, 2)) + '%', (data_matrix[0], data_matrix[1] + 60), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                     (0, 255, 0), 2)

     cv2.imshow('Output',img)
     cv2.waitKey(1)