import numpy as np
import cv2 

cam = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

# Load data in train (X and y)

face01 = np.load("face_01.npy").reshape((20,-1))
face02 = np.load("face_02 (1).npy").reshape((20,-1))
face03 = np.load("face_03.npy").reshape((20,-1))

data = np.concatenate((face01,face02,face03))
labels = np.zeros((data.shape[0]))

labels[20:40] = 1.0
labels[40:] = 2.0 

names = {
    0: 'Navneet',
    1: 'Parth',
    2: 'Prateek'
}

# @desc Define the KNN Functions

# @desc Function to compute the euclidean distance

def distance(x1,x2):
    d = np.sqrt(((x1-x2)**2).sum())
    return d 

# @desc Function KNearestNeighbours

def knn(X_train,Y_train,xt,k=5):
    vals = []
    for ix in range(X_train.shape[0]):
        d = distance(X_train[ix],xt)
        vals.append([d,Y_train[ix]])
    sorted_labels = sorted(vals,key=lambda z: z[0])
    neighbours = np.asarray(sorted_labels)[:k,-1]

    freq = np.unique(neighbours,return_counts=True)

    return freq[0][freq[1].argmax()]

    # @desc Face Detection Loop
while True:
    ret, frame = cam.read()
    if ret==True:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            # Extract the detected face
            face = frame[y:y+h,x:x+w]
            #Resize to a fixed shape
            resized_face = cv2.resize(frame,(50,50)).flatten()
            text = names[int(knn(data, labels, resized_face))]
            cv2.putText(frame,text,(x,y),font,1,(255,255,0),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        cv2.imshow('Frame',frame)

        if cv2.waitKey(1) == 27:
            break
        
    else:
        print('Error')
        break


cv2.destroyAllWindows()
