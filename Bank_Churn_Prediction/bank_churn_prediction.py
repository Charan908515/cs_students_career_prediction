import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
data=pd.read_csv("C:/Users/nchar/Downloads/Churn_Modelling.csv")
df=pd.DataFrame(data)
df.drop(columns=["RowNumber","CustomerId","Surname"],axis=1,inplace=True)
print(df.head())
print(df.info())
print(df.isnull().sum())
map1={"Male":0,"Female":1}
map2={"Spain":0,"France":1,"Germany":2}
df["Geography"]=df["Geography"].map(map2)
df["Gender"]=df["Gender"].map(map1)
print(df.info())
print(df["Geography"].unique())
print(df["Gender"].unique())
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x=df.drop(columns=["Exited"],axis=1)
y=df["Exited"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
scaler=StandardScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)
model=Sequential([
    Dense(20,activation="relu",input_shape=(xtrain.shape[1],)),
    Dense(10,activation="relu"),
    Dense(1,activation="sigmoid")])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
callback=EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)
model.fit(xtrain,ytrain,callbacks=[callback],epochs=100)
#map1={"Male":0,"Female":1}
#map2={"Spain":0,"France":1,"Germany":2}

input1=np.array([465,1,1,51,8,122522.32,1,0,0,181297.65])
input1=scaler.transform([input1])
prediction1=model.predict(input1)
print(prediction1)
if prediction1<=0.5:
    print("no")
else:
    print("yes")
input2=np.array([556,1,1,61,2,117419.35,1,1,1,94153.83])
input2=scaler.transform([input2])
prediction2=model.predict(input2)
print(prediction2)
if prediction2<=0.5:
    print("no")
else:
    print("yes")
st.title("Bank Churn Prediction")
credit_score=st.slider("enter credit score",400,1000,651)
geography=st.selectbox("choose the region",map2.keys())
gender=st.selectbox("choose the region",map1.keys())
age=st.slider("enter age",18,100,21)
tenure=st.slider("enter tenure",0,10)
balance=st.slider("enter balance",0.0,10000.2)
nop=st.slider("enter nop",0,10)
cr=st.slider("has card",0,1)
active=st.slider("is active",0,1)
geography=map2[geography]
gender=map1[gender]
input3=np.array([credit_score,geography,gender,age,tenure,balance,nop,cr,active,94153.83])
input3=scaler.transform([input3])
prediction3=model.predict(input3)
print(prediction3)
if prediction2<=0.5:
    st.write("no")
else:
    st.write("yes")


