import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.image import rgb_to_grayscale
from tensorflow.compat.v1 import *

video = './test_video.MP4'

font = cv2.FONT_HERSHEY_SIMPLEX
cap1 = cv2.VideoCapture(video)
cap2 = cv2.VideoCapture(video)
cap3 = cv2.VideoCapture(video)
cap4 = cv2.VideoCapture(video)

width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('video4.mp4', fourcc, 20.0, (640, 360))

while(cap1.isOpened()):
    
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    ret, frame3 = cap3.read()
    ret, frame4 = cap4.read()
    
    if ret == True:
        
        #########Original##########x

        frame1 = cv2.resize(frame1, (width//2, height//2))
        
        cv2.putText(frame1,'Original',(10,30), font, 1, (0,0,255),1)
        
        
        #####Gaussian filtering####
        
        frame2 = cv2.resize(frame2, (width//2, height//2))
        
        noisy_image_9 = (frame2.astype(np.float) + np.random.randn(*frame2.shape)*9).astype(dtype=np.uint8)
        result2 = cv2.GaussianBlur(noisy_image_9,(0,0),5)

        cv2.putText(result2,'Gaussian filtering',(10,30), font, 1, (0,0,255),1)
        
           
        ####Histogram equalization####

        frame3 = cv2.resize(frame3, (width//2, height//2))
        
        hsv_image = cv2.cvtColor(frame3,cv2.COLOR_BGR2HSV)
        hsv_image[:,:,2] = cv2.equalizeHist(hsv_image[:,:,2])
        histEqualized_image = cv2.cvtColor(hsv_image,cv2.COLOR_HSV2BGR)

        cv2.putText(histEqualized_image,'Histogram equalization',(10,30), font, 1, (0,0,255),1)
        
        result3 = histEqualized_image
        
        ###########Gray#############

        frame4 = cv2.resize(frame4, (width//2, height//2))
        
        disable_eager_execution() 
        config=ConfigProto()
        config.gpu_options.allow_growth = True 
        sess = Session(config=ConfigProto()) 
        image_data = sess.run(rgb_to_grayscale(frame4))
        result4 = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)

        cv2.putText(result4,'Gray',(10,30), font, 1, (0,0,255),1)
        
        ###########################
        
        up_frame = np.hstack((frame1, result2))
        down_frame = np.hstack((result3, result4))
        result_video = np.vstack((up_frame, down_frame))
        
        out.write(result_video)
    else:
        break
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()