import cv2
from tensorflow.keras.models import load_model
import numpy as np

face_detection_model=cv2.CascadeClassifier('model\haarcascade_frontalface_default.xml')
emotion_detection_model=load_model('model\emotion_detector2.h5')
emotion_list=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
cam=cv2.VideoCapture(0)
while True:
    ret, frame=cam.read()

    gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates=face_detection_model.detectMultiScale(gray_frame)

    
    for x,y,width,height in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+width,y+height),(0,0,255),2)
        crop_detected_face=gray_frame[y:y + height, x:x + width]
        resized_frame = np.expand_dims(np.expand_dims(cv2.resize(crop_detected_face, (48, 48)), -1), 0)

        emotion=emotion_detection_model.predict(resized_frame)
        maxindex = int(np.argmax(emotion))
        cv2.putText(frame, emotion_list[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  

cam.release()

cv2.destroyAllWindows()