import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
d=pd.read_csv("C:/Users/nchar/Documents/pythonProject/anemia_prediction/anemia.csv")
df=pd.DataFrame(d)
df.drop(columns=["Number"],axis=1,inplace=True)
print(df["Anaemic"].value_counts())
l1=LabelEncoder()
l2=LabelEncoder()
df["Sex"]=l1.fit_transform(df["Sex"])
df["Anaemic"]=l2.fit_transform(df["Anaemic"])
print(df.info())
print(df.describe())
sns.heatmap(df.corr(),annot=True)
sns.pairplot(df)
x=df.drop(columns=["Anaemic"],axis=1)
y=df["Anaemic"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)
model=LogisticRegression()
model.fit(xtrain,ytrain)
print("accuracy of trainning data :",accuracy_score(ytrain,model.predict(xtrain)))
print("accuracy of test data :",accuracy_score(ytest,model.predict(xtest)))
new_input=np.array(["M",43.2555,30.8421,25.9025,6.3]).reshape(1,5)
new_data=pd.DataFrame(new_input,columns=x.columns)
new_data["Sex"]=l1.transform(new_data["Sex"])
print("output :",model.predict(new_data))
