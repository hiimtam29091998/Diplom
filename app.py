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
calories = pd.read_csv('C:/Users/Thanh Tung/PycharmProjects/Diplom_Tam/calories.csv')
exercise = pd.read_csv('C:/Users/Thanh Tung/PycharmProjects/Diplom_Tam/exercise.csv')
Data = pd.concat([exercise, calories['Calories']], axis=1)
Data.replace({"Gender":{'male':0,'female':1}}, inplace=True)


X = Data.drop(columns=['User_ID','Calories'], axis=1).values
Y = Data['Calories'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#Нормализация данных в интервале [0, 1]
std = MinMaxScaler()
std.fit(X_train)
#X_train = std.transform(Xtrain)
#X_test = std.transform(Xtest)

model = load_model('my_model.h5')

# Tạo ứng dụng
app = Flask(__name__)

# Liên kết hàm home với URL
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def logo():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(i) for i in request.form.values()]
    X = np.array(features)
    if features[0] == 0:
        features[0] = 'Male'
    else:
        features[0] = 'Female'
    X = X.reshape(1, 7)
    X = std.transform(X)
    #Predict
    prediction = model.predict(X)
    output = int(prediction[0, 0])
    #Kiểm tra các giá trị đầu ra và truy xuất kết quả bằng thẻ html dựa trên giá trị
    return render_template('result.html', gender = features[0],
                           age = features[1],
                           height = features[2],
                           weight = features[3],
                           duration = features[4],
                           heart_rate = features[5],
                           body_temp = features[6],
                           result = "The Calories burnt for the entered details is {} kcal".format(output))

if __name__ == '__main__':
    #Chạy ứng dụng
    app.run()