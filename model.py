import pickle
import numpy as np
import pandas as pd
import matplotlib_inline as plt
df=pd.read_csv("Crop_recommendation.csv")

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

pickle.dump(model,open("model.pkl","wb"))
