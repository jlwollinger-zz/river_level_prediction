# -*- coding: utf-8 -*-


import json
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_left_and_right_means(mask):
    means_left = []
    iteration_count = 0
    intersection_points = []
    for i in range(0, len(mask), 1):
       for j in range(len(mask[i])):
           if mask[i][j] == True:
               intersection_points.append(j)
               break
       iteration_count = iteration_count + 1     
       if iteration_count == 10:
           if len(intersection_points) > 0:
               means_left.append(sum(intersection_points) / len(intersection_points))
           else:
               means_left.append(0)
           iteration_count = 0
           intersection_points = []

    means_right = []
    iteration_count = 0
    intersection_points = []
    for i in range(len(mask) -1, -1, -1):
       for j in range(len(mask[i])):
           if mask[i][j] == True:
               intersection_points.append(j)
               break
       iteration_count = iteration_count + 1     
       if iteration_count == 10:
           if len(intersection_points) > 0:
               means_right.append(sum(intersection_points) / len(intersection_points))
           else:
               means_right.append(0)
           iteration_count = 0
           intersection_points = []
           
    means_left.extend(means_right)
    return means_left
    

X_train = np.array([])
y_train = np.array([])

annotations = json.load(open(os.path.join('./dataset2/train/', "annotations.json")))

values = list(annotations.values())
annotations = [a for a in values if a['regions']]

pbar = tqdm(annotations)
for value in pbar: 
    current_file = value
    y = float(current_file['filename'].split('_')[0])
    regions = current_file['regions']
    if regions != None and regions['0']['shape_attributes']['all_points_x'] != None:
        
        all_x = regions['0']['shape_attributes']['all_points_x']
    
        all_y = regions['0']['shape_attributes']['all_points_y']
    
    #plt.scatter(all_x, all_y)
    #plt.show()
    
        all_x = np.array(all_x)
        all_y = np.array(all_y)
    
        xy = np.dstack((all_x.ravel(), all_y.ravel()))[0]
    
    
        mask = np.zeros((1080, 1920))
        cv.fillPoly(mask, np.int32([xy]), 1)
        mask = mask.astype(bool)
    
    #plt.imshow(mask)
    
    
        x = get_left_and_right_means(mask)
    
        X_train = np.append(X_train, x)
        y_train = np.append(y_train, y)
    
    
X_train = X_train.reshape((len(y_train), int(len(X_train) / len(y_train))))
#X_train, y_train = np.array(X_train), np.array(y_train)

#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#Testar com outras...
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Melhor resultado foi BayesianRidge

regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)


score = regressor.score(X_test, y_test)
print(score)


#PLOT
from sklearn.metrics import mean_squared_error, mean_absolute_error
test_mse = mean_squared_error(y_test, regressor.predict(X_test))

print("MSE " + str(test_mse))
print('Coefficients: \n', regressor.coef_) 
  
print('Variance score: {}'.format(regressor.score(X_test, y_test))) 

#plt.style.use('fivethirtyeight') 
  
#plt.scatter(regressor.predict(X_train), regressor.predict(X_train) - y_train, 
#            color = "green", s = 2, label = 'Train data') 
  
#plt.scatter(regressor.predict(X_test), regressor.predict(X_test) - y_test, 
#            color = "blue", s = 2, label = 'Test data') 
  
#plt.hlines(y = 0, xmin = 0, xmax = 2, linewidth = 2) 
#plt.legend(loc = 'upper right') 
#plt.title("Residual errors") 
#plt.show() 


## The line / model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.plot(predictions, regressor.predict(X_test), color='red')
plt.show() 


##Dumping
from sklearn.externals import joblib
joblib.dump(regressor, 'regressor.sav')

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

teste = X_train.reshape((1291, 216, 1))
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(Dense(50, input_dim=86, activation='relu'))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(Dense(50, input_dim=20, activation='relu'))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(Dense(25, input_dim=20, activation='softmax'))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


test = regressor.predict(X_test)

#Teste
X_test = np.array([])
y_test = np.array([])

annotations = json.load(open(os.path.join('./dataset/test/', "annotations.json")))

values = list(annotations.values())
annotations = [a for a in values if a['regions']]

pbar = tqdm(annotations)
for value in pbar:  
    current_file = value
    y = float(current_file['filename'].split('_')[0])
    regions = current_file['regions']
    if regions != None and regions['0']['shape_attributes']['all_points_x'] != None:
        
        all_x = regions['0']['shape_attributes']['all_points_x']
    
        all_y = regions['0']['shape_attributes']['all_points_y']
    
        all_x = np.array(all_x)
        all_y = np.array(all_y)
    
        xy = np.dstack((all_x.ravel(), all_y.ravel()))[0]
    
    
        mask = np.zeros((1080, 1920))
        cv.fillPoly(mask, np.int32([xy]), 1)
        mask = mask.astype(bool)

        x = get_left_and_right_means(mask)
    
        X_test = np.append(X_test, x)
        y_test = np.append(y_test, y)
    
    
X_test = X_test.reshape((len(y_test), int(len(X_test) / len(y_test))))

