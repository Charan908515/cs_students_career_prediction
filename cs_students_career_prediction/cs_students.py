import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
import pickle
data=pd.read_csv("C:/Users/nchar/Documents/pythonProject/cs_students_career_prediction/cs_students.csv")
df=pd.DataFrame(data)
df.drop(columns=["Student ID","Name","Major"],inplace=True,axis=1)
print(df.info())
print(df.head())
obj=[colum for colum in df.columns if df[colum].dtype=="O"]
columns=["Gender","Age","GPA","Interested Domain","Projects","Python","SQL","Java"]
print(obj)
for i in obj:
    print(len(df[i].unique()))
gender_encoder=LabelEncoder()
df["Gender"]=gender_encoder.fit_transform(df["Gender"])
domain_encoder=LabelEncoder()
df["Interested Domain"]=domain_encoder.fit_transform(df["Interested Domain"])
project_encoder=LabelEncoder()
df["Projects"]=project_encoder.fit_transform(df["Projects"])
future_encoder=LabelEncoder()
df["Future Career"]=future_encoder.fit_transform(df["Future Career"])
py_encoder=LabelEncoder()
df["Python"]=py_encoder.fit_transform(df["Python"])
sql_encoder=LabelEncoder()
df["SQL"]=sql_encoder.fit_transform(df["SQL"])
java_encoder=LabelEncoder()
df["Java"]=java_encoder.fit_transform(df["Java"])
print(df.info())
x=df.drop(columns=["Future Career"])
y=df["Future Career"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
scaler=StandardScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)
model=GradientBoostingClassifier()
model.fit(xtrain,ytrain)
print(accuracy_score(ytrain,model.predict(xtrain)))
print(accuracy_score(ytest,model.predict(xtest)))
print(classification_report(ytest,model.predict(xtest)))
input_data=np.array([["Female",21,3.7,"Cloud Computing","AWS Deployment","Weak","Strong","Average"]])
input_data=pd.DataFrame(input_data,columns=columns)
input_data["Gender"]=gender_encoder.transform(input_data["Gender"])
input_data["Interested Domain"]=domain_encoder.transform(input_data["Interested Domain"])
input_data["Projects"]=project_encoder.transform(input_data["Projects"])
input_data["Python"]=py_encoder.transform(input_data["Python"])
input_data["SQL"]=sql_encoder.transform(input_data["SQL"])
input_data["Java"]=java_encoder.transform(input_data["Java"])
input_data=scaler.transform(input_data)
print((model.predict(input_data)))
pickle.dump(scaler,open("cs_students_scaler.pkl","wb"))
pickle.dump(gender_encoder,open("cs_students_gender_encoder.pkl","wb"))
pickle.dump(domain_encoder,open("cs_students_domain_encoder.pkl","wb"))
pickle.dump(project_encoder,open("cs_students_project_encoder.pkl","wb"))
pickle.dump(py_encoder,open("cs_students_py_encoder.pkl","wb"))
pickle.dump(sql_encoder,open("cs_students_sql_encoder.pkl","wb"))
pickle.dump(java_encoder,open("cs_students_java_encoder.pkl","wb"))
pickle.dump(model,open("cs_students_model.pkl","wb"))
pickle.dump(future_encoder,open("cs_students_future_career.pkl","wb"))
