import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
d=pd.read_csv("C:/Users/nchar/Documents/pythonProject/diabetis-prediction/diabetes.csv")
data=pd.DataFrame(d)
data.info()
sns.heatmap(data.corr())
y=data["Outcome"]
x=data.drop(["Outcome"],axis=1)
s=StandardScaler()
s.fit(x)
x=s.transform(x)
print(x.shape)
xtrain,xtest,ytrain,ytest=train_test_split(x,y)
l=LogisticRegression()
l.fit(xtrain,ytrain)
xtrain1pre=l.predict(xtrain)
xtest1pre=l.predict(xtest)
print(accuracy_score(ytest,xtest1pre))
print(confusion_matrix(ytest,xtest1pre))
er=np.array([[7,107,74,0,0,29.6,0.254,31]])
print(er.shape)
qw=s.transform(er)
print(l.predict(qw))
st.title("DIABETIS PREDICTION")
pregnency=st.slider("Pregnancies:",data["Pregnancies"].min(),data["Pregnancies"].max())
glucose=st.slider("glucose level:",data["Glucose"].min(),data["Glucose"].max())
bp=st.slider("BloodPressure:",data["BloodPressure"].min(),data["BloodPressure"].max())
skin=st.slider("SkinThickness:",data["SkinThickness"].min(),data["SkinThickness"].max())
insulin=st.slider("Insulin:",data["Insulin"].min(),data["Insulin"].max())
bmi=st.slider("BMI:",data["BMI"].min(),data["BMI"].max())
dpf=st.slider("DiabetesPedigreeFunction:",data["DiabetesPedigreeFunction"].min(),data["DiabetesPedigreeFunction"].max())
age=st.slider("age:",data["Age"].min(),data["Age"].max())
n=np.array([[pregnency,glucose,bp,skin,insulin,bmi,dpf,age]])
na=s.transform(n)
print(l.predict(na)[0])
predicted_value=l.predict(na)
st.write("prediction")
st.write(f"the person has sugar if the value is zero : {predicted_value[0]}")




