import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import load_model

#Собрать данные
calories = pd.read_csv('C:/Users/Thanh Tung/PycharmProjects/Diplom_Trung/calories.csv')
exercise = pd.read_csv('C:/Users/Thanh Tung/PycharmProjects/Diplom_Trung/exercise.csv')
Data = pd.concat([exercise, calories['Calories']], axis=1)
Data.replace({"Gender":{'male':0,'female':1}}, inplace=True)


X = Data.drop(columns=['User_ID','Calories'], axis=1).values
Y = Data['Calories'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
#Нормализация данных в интервале [0, 1]
std = MinMaxScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)

#Скремблирование данных
perms_ = np.random.permutation(X_train.shape[0])
X_train = X_train[perms_]
Y_train = Y_train[perms_]

#Linear Regression with  Multi-layer Perceptron
model = keras.Sequential()
model.add(keras.Input(shape = (7,)))
model.add(keras.layers.Dense(10, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(keras.layers.Dense(10, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(keras.layers.Dense(10, activation = 'relu', kernel_initializer = 'he_uniform'))

model.add(keras.layers.Dense(1, activation = 'linear'))

optimizer = keras.optimizers.SGD(learning_rate = 0.001, momentum=0.9)
model.compile(optimizer = optimizer, loss = 'mean_absolute_error')
history = model.fit(X_train, Y_train, epochs = 200, batch_size = 256, validation_data = (X_test, Y_test), verbose = 2)

test_mae = model.evaluate(X_test, Y_test, verbose=0)
print('Test error: %.4f' %test_mae)

model.save('my_model.h5')

#model = load_model('my_model.h5')