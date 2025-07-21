from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model=Sequential()
model.add(Dense(units=2,activation='sigmoid'))
model.add(Dense(units=1,activation='sigmoid')) #

model.summary()

model.compile(optimizer='lion',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X,y,epochs=5000,verbose=0)

result=model.predict(X)

for i in result:
  if i>0.90:
    print("1")
  else:
    print("0")

