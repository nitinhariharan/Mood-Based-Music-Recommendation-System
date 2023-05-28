import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import asarray
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model_1.h5')

import joblib
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from youtubesearchpython.__future__ import VideosSearch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import asyncio
from deepface import DeepFace
import pickle

data = pd.read_csv("data.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')
print(data.head())

"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10,))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)

from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)


projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']
"""

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False))
                                 ], verbose=False)
                                 
X = data.select_dtypes(np.number)
#number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
pickle.dump(song_cluster_pipeline, open('song_cluster_pipeline.sav', 'wb'))

#song_cluster_pipeline = pickle.load(open('song_cluster_pipeline.sav', 'rb'))
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

#song_cluster_pipeline
"""
from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)

projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']
"""

from imutils import face_utils
import numpy as np
import dlib
import cv2

def predict_value(img1):
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(img1, (48, 48)), -1), 0)
        # to display the face numbe
        
        a=[]
        #cv2.imwrite("user_me.jpg",cropped_img)

        a.append(int(np.argmax(emotion_model.predict(cropped_img))))
        print(a)
        b = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
        for i in range(0,len(a)):
                Tv = DeepFace.analyze(img_path = "user.jpg", 
                actions =  'emotion',enforce_detection=False
                )
                Tv=Tv[0]['emotion']
                Keymax = max(zip(Tv.values(), Tv.keys()))[1]
                print(Keymax)
                return Keymax
        
def main_data_of_face(x1):
    a=[]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    img_path=x1
    img = cv2.imread('user.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    Tv = DeepFace.analyze(img_path = "user.jpg", 
                actions =  'emotion',enforce_detection=False
    )
    Tv=Tv[0]['emotion']
    Keymax = max(zip(Tv.values(), Tv.keys()))[1]
    

    for (j,face) in enumerate(faces):
            points = predictor(gray, face)

            print('\nface #',j+1)
            l=[]
            print('\nface boundary coordinate\n')
            for i in range(0, 27):  # loop for displaying face boundary coordinate
                curr_c = (points.part(i).x, points.part(i).y)
            #print(curr_c)
        #print('\nnose coordinate\n')
            for i in range(27, 36):  # loop for displaying nose  coordinate
                curr_c = (points.part(i).x, points.part(i).y)
            #print(curr_c)
        #print('\nleft eye coordinate\n')
            for i in range(36, 42):  # loop for displaying left eye coordinate
                curr_c = (points.part(i).x, points.part(i).y)
            #print(curr_c)
        #print('\nright eye coordinate\n')
            for i in range(42, 48):  # loop for displaying right eye coordinate
                curr_c = (points.part(i).x, points.part(i).y)
            #print(curr_c)
        #print('\nlips coordiante\n')
            for i in range(48, 68):  # loop for displaying lips coordinate
                curr_c = (points.part(i).x, points.part(i).y)
            #print(curr_c)

            for i in range(5, 12):                          #loop for storing jaw coordinates
                curr_c=(points.part(i).x, points.part(i).y)
                l.append(curr_c)

            #cv2.line(img, curr_c, next_cordi, (0, 0, 0), 3)
            for n in range(0, 68):                          #loop for detecting feature points on face
                x = points.part(n).x
                y = points.part(n).y
            #cv2.circle(img, (x, y), 3, (0, 0, 255), 2)

        #points = face_utils.shape_to_np(points)

        # to  convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img1=img[y:y+h,x:x+w]
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(img1, (48, 48)), -1), 0)
        # to display the face numbe
                
        #cv2.imwrite("user_me.jpg",cropped_img)

            a.append(int(np.argmax(emotion_model.predict(cropped_img))))
            b = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
            for i in range(0,len(a)):
                Tv = DeepFace.analyze(img_path = "user.jpg", 
                actions =  'emotion',enforce_detection=False
                )
                Tv=Tv[0]['emotion']
                Keymax = max(zip(Tv.values(), Tv.keys()))[1]
                return Keymax
    return Keymax
            
        