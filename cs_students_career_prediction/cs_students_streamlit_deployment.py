import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import numpy as np
import warnings
import sys
import path
# Suppress specific warning
warnings.filterwarnings("ignore", message="X does not have valid feature names")
dir = path.Path(__file__).abspath()
sys.append.path(dir.parent.parent)

# load model
scaler = "./cs_students_career_prediction/cs_students_scaler.pkl"


with open(scaler, 'rb') as file:
    scaler = pickle.load(file)
    

model = "./cs_students_career_prediction/cs_students_model.pkl"


with open(model, 'rb') as file:
    model = pickle.load(file)
gender_encoder = "./cs_students_career_prediction/cs_students_gender_encoder.pkl"


with open(gender_encoder, 'rb') as file:
    gender_encoder = pickle.load(file)
java_encoder = "./cs_students_career_prediction/cs_students_java_encoder.pkl"


with open(java_encoder, 'rb') as file:
    java_encoder = pickle.load(file)
py_encoder = "./cs_students_career_prediction/cs_students_py_encoder.pkl"


with open(py_encoder, 'rb') as file:
    py_encoder = pickle.load(file)
sql_encoder = "./cs_students_career_prediction/cs_students_sql_encoder.pkl"


with open(sql_encoder, 'rb') as file:
    sql_encoder = pickle.load(file)
domain_encoder = "./cs_students_career_prediction/cs_students_domain_encoder.pkl"


with open(domain_encoder, 'rb') as file:
    domain_encoder = pickle.load(file)
project_encoder = "./cs_students_career_prediction/cs_students_project_encoder.pkl"


with open(project_encoder, 'rb') as file:
    project_encoder = pickle.load(file)
model=pickle.load(open("Charan908515/ml_projects/cs_students_career_prediction/cs_students_model.pkl","rb"))
gender_encoder=pickle.load(open("Charan908515/ml_projects/cs_students_career_prediction/cs_students_gender_encoder.pkl","rb"))
java_encoder=pickle.load(open("Charan908515/ml_projects/cs_students_career_prediction/cs_students_java_encoder.pkl","rb"))
sql_encoder=pickle.load(open("Charan908515/ml_projects/cs_students_career_prediction/cs_students_sql_encoder.pkl","rb"))
py_encoder=pickle.load(open("Charan908515/ml_projects/cs_students_career_prediction/cs_students_py_encoder.pkl","rb"))
domain_encoder=pickle.load(open("Charan908515/ml_projects/cs_students_career_prediction/cs_students_domain_encoder.pkl","rb"))
project_encoder=pickle.load(open("Charan908515/ml_projects/cs_students_career_prediction/cs_students_project_encoder.pkl","rb"))
with open(".cs_students_career_prediction/cs_students_future_career.pkl", "rb") as file:
    future_encoder = pickle.load(file, encoding="latin1")
data=pd.read_csv("./cs_students_career_prediction/cs_students.csv")
df=pd.DataFrame(data)
domain_options=df["Interested Domain"].unique()
project_options=df["Projects"].unique()
st.title("CS STUDENTS CAREER PREDICTION")
options=["Strong","Weak","Average"]
gender_options=["Female","Male"]
gender=st.selectbox("Choose your gender",gender_options)
age=[st.slider("select your age:",0,100,25)]
gpa=[st.slider("select your gpa:",0.0,10.0,2.5)]
domain=st.selectbox("choose the domain you are interested:",domain_options)
project=st.selectbox("choose the projects you did:",project_options)
python=st.selectbox("choose your level in Python:",options)
sql=st.selectbox("choose your level in SQL:",options)
java=st.selectbox("choose your level in Java:",options)




gender=gender_encoder.transform([gender])
domain=domain_encoder.transform([domain])
project=project_encoder.transform([project])
python=py_encoder.transform([python])
sql=sql_encoder.transform([sql])
java=java_encoder.transform([java])

input_data=np.array([gender,age,gpa,domain,project,python,sql,java])
print(input_data.shape)
input_data=input_data.reshape([1,8])
print(input_data.shape)

input_data=scaler.transform(input_data)
pre=future_encoder.inverse_transform(model.predict(input_data))
print(pre[0])
st.write(f"You can choose the {pre[0]} as your career")
