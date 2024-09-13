import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime
from sklearn.preprocessing import StandardScaler
import re
from sklearn.model_selection import train_test_split

d=pd.read_csv("C:/Users/nchar/Documents/pythonProject/diabetis-prediction/diabetes.csv")
data=pd.DataFrame(d)
data.info()
y=data["Outcome"]
x=data.drop(["Outcome"],axis=1)
s=StandardScaler()
s.fit(x)
x=s.transform(x)
print(x.shape)
xtrain,xtest,ytrain,ytest=train_test_split(x,y)
print(xtrain.shape)
print(ytrain.shape)

# creating the network
model=Sequential(
    [
        Dense(64,activation="relu",input_shape=(xtrain.shape[1],)),
        Dense(32,activation="relu"),
        Dense(1,activation="sigmoid")
    ]
)  
# summarizing/finalizing the network
model.summary()

#giving the optimizers and the loss function and also the metrics and compile/apply these these to the network  
adam=tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=adam,loss="binary_crossentropy",metrics=["accuracy"])

#creating the the log files

#tensorboard is used to visualize the gradient descent and the also many more it will save the visualzation or data about the network every time we run the code(every time we fit the data) 
log_dir="log/fit"
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

# early stopping is used to terminate the trainning if the model is already trained effectively without completion of totsl no.of epochs
earlystopping=EarlyStopping(monitor="val_loss",patience=30,restore_best_weights=True)

history=model.fit(
    xtrain,ytrain,validation_data=(xtest,ytest),epochs=100,
    callbacks=[tensorflow_callback,earlystopping]
)

model.save("model.h5")



