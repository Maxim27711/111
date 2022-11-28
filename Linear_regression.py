import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from keras.layers import Activation, Dense, Flatten
import numpy as np
from keras import layers
from keras import models
import tensorflow as tf
from tensorflow.keras.models import Sequential
#Первичная Обработка Dataframe:

#считываем файл в DataFrame
df = pd.read_csv("ebw_data.csv")
#print(df.describe())
#Расммотрим корреляционную зависимость
#print(df.head())
#print(df.corr(method='pearson'))
#разделяем DataFrame на входные и искомые данные
df_Param = df[['Depth','Width']]
df_data = df.drop(['Depth','Width'], axis=1)
#Делим данные на тренировочные и тестируемые
Xtrn, Xtest, Ytrn, Ytest = train_test_split(df_data, df_Param, test_size=0.4)
#Применяем метод линейной регрессии
regr=LinearRegression().fit(Xtrn,Ytrn)
y_pred = regr.predict(Xtest)
#Оценка модели
#print("Оценка модели", regr.score(Xtest,Ytest))
#Вводим дополнительные значения на тест:
#new_test = pd.DataFrame([[45,140,8.5,75]], columns=['IW','IF','VW','FP'])
#New_pred = regr.predict(new_test)
#print(New_pred)

#           Стандартизация данных
# Среднее значение
mean = Xtrn.mean(axis=0)
# Стандартное отклонение
std = Xtrn.std(axis=0)
Xtrn -= mean
Xtrn /= std
Xtest -= mean
Xtest /= std

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(Xtrn.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))
print(model.summary())

model.compile(optimizer='nadam', loss='mse', metrics=['mae'])
history = model.fit(Xtrn,
                    Ytrn,
                    epochs=300,
                    validation_split=0.1,
                    verbose=2)

plt.plot(history.history['mae'],
         label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(history.history['val_mae'],
         label='Средняя абсолютная ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

predictions = model.predict(Xtest)
