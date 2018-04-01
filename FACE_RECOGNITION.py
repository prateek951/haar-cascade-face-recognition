import numpy as np
import cv2 

cam = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

# Load data in train (X and y)

face01 = np.load("face_01.npy").reshape((20,-1))
face02 = np.load("face_02.npy").reshape((20,-1))
face03 = np.load("face_03.npy").reshape((20,-1))

data = np.concatenate((face01,face02,face03))
labels = np.zeros((data.shape[0]))

labels[20:40] = 1.0
labels[40:] = 2.0 

names = {
    0: 'Prateek',
    1: 'Lorem',
    2: 'Sachin'
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



