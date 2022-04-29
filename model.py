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

from flask import Flask, request, render_template

#Собрать данные
calories = pd.read_csv('C:/Users/Thanh Tung/PycharmProjects/Diplom_Trung/calories.csv')
exercise = pd.read_csv('C:/Users/Thanh Tung/PycharmProjects/Diplom_Trung/exercise.csv')
Data = pd.concat([exercise, calories['Calories']], axis=1)
Data.replace({"Gender":{'male':0,'female':1}}, inplace=True)


X = Data.drop(columns=['User_ID','Calories'], axis=1).values
Y = Data['Calories'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X_test[0:1])
#Нормализация данных в интервале [0, 1]
std = MinMaxScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)


model = load_model('my_model.h5')


output = model.predict(X_test[0:1])
print(output)