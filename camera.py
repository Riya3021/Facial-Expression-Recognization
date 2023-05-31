from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import load_img, img_to_array
#from keras.preprocessing.image import img_to_array
#from tensorflow.keras.utils import load_img
#from keras.preprocessing import image

import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier(r'C:\Users\inspiron 5501\Desktop\cpy\haarcascade_frontalface_alt2.xml')
classifier =load_model(r'C:\Users\inspiron 5501\Desktop\cpy\model.h5')

emotion_labels = ['Angry','disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret,frame=self.video.read()
        faces=faceDetect.detectMultiScale(frame, 1.3, 5)
       
        while True:
            _, frame = self.video.read()
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray)

            for (x,y,w,h) in faces:
                x1,y1=x+w, y+h
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)

                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                    prediction = classifier.predict(roi)[0]
                    label=emotion_labels[prediction.argmax()]
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                    
                    ret,jpg=cv2.imencode('.jpg',frame)
                    return jpg.tobytes()

                else:
                    cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()
            
            
    