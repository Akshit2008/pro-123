import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

X=np.load('image.npz')['arr_0']
y=pd.read_csv("label.csv")["labels"]
print(pd.series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

samples_per_class = 5
figure=plt.figure(figsize=(nclasses*2,(1+samples_per_class*2)))

idx_cls=0
for cls in classes:
    idxs=np.flatnonzero(y==cls)
    idxs=np.random.choice(idxs,samples_per_class,replace=False)
    i=0
    for idx in idxs:
        plt_idx=i*nclasses + idx_cls +1
        p=plt.subplot(samples_per_class,nclasses,plt_idx);
        p=sns.heatmap(np.reshape(X[idx],(22,30)),cmap=plt.cm.gray,
                      xticklabels=False,yticklabels=False,cbar=False);
        p=plt.axis('off');
        i+=1
    idx_cls+=1

x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
X_train_scaled=x_train/255
X_test_scaled=x_test/255