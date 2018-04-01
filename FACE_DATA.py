# Collect the data

import numpy as np
import cv2 

web_cam = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

data = []
ix = 0


while True:
    # Read the slice of the video i.e a frame
    ret,frame = web_cam.read()
    # Check for frame
    if ret == True:
        # Got the frame
        # Convert the frame to the grayscale
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Grab all the faces in the grayscaled frame
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        # Cycle through the faces
        for (x,y,w,h) in faces:
            face_section = faces[y:y+h,x:w]
            # Resize the face_section
            resized_face = cv2.resize(face_section,(50,50))
            if ix%10 == 0 and len(data)<20:
                data.append(resized_face)
            #Trace the rectangle over the face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        ix += 1
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1)==27 or len(data) >=20:
            break
    else:
        print("Error occured")
        break 

cv2.destroyAllWindows()
data = np.asarray(data)

print(data.shape)
np.save('face03',data)
    
