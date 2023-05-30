#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn


# In[4]:


datos_hoy=pd.read_csv('datosbeta.csv')


# In[5]:


datos_hoy


# In[6]:


fechas_hoy=pd.read_csv('fechasbeta.csv')


# In[7]:


data=pd.concat([fechas_hoy,datos_hoy], axis = 1)


# In[8]:


data


# In[9]:


data['datetime']=pd.to_datetime(data['Fecha'],format='%d.%m.%Y')


# In[10]:


df2=data['datetime']


# In[11]:


df2


# In[12]:


newdata=pd.concat([df2,datos_hoy], axis = 1)


# In[13]:


newdata


# In[14]:


newdata.set_index('datetime')


# In[15]:


realnewdata=newdata.set_index('datetime')


# In[16]:


realnewdata


# In[17]:


realnewdata.resample("M").mean()


# In[18]:


df3=realnewdata.resample("M").mean()


# In[19]:


df3.plot(kind='line')


# In[20]:


df3


# In[21]:


# Set entrenamiento y set validación
set_entrenamiento = df3[:'2022'].iloc[:,0]
set_validacion = df3['2023':].iloc[:,0]


# In[22]:


set_entrenamiento.plot(legend=True)
set_validacion.plot(legend=True)
plt.legend(['Entrenamiento (2012-2022)', 'Validación (2023)'])
plt.show()


# In[23]:


df4=set_entrenamiento.values.reshape(-1,1)


# In[24]:


sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(df4)


# In[25]:


time_step = 60
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)


# In[26]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[27]:


# Red LSTM
#
dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 50

modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse')
modelo.fit(X_train,Y_train,epochs=20,batch_size=32)


# In[28]:


df4.reshape(1,-1)


# In[29]:


df4


# In[30]:


df6=set_validacion.values.reshape(1,-1)


# In[31]:


df7=set_validacion.values.reshape(-1,1)


# In[32]:


set_validacion.values.reshape(-1,1)


# In[33]:


# Validación (predicción del valor de las acciones)
#
x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)

