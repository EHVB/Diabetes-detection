import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder    
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as a_s 

#import dataset
data_set= pd.read_csv('data.csv')

#extracting dependent and independent variables
x= data_set.iloc[:,1:-1].values  
y= data_set.iloc[:,-1].values  

#encoding categorical data
label_encoder_x= LabelEncoder()
x[:, 5]= label_encoder_x.fit_transform(x[:, 5])

labelencoder_y= LabelEncoder()
y= labelencoder_y.fit_transform(y)   

#splitting the Dataset into the training set and test set
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.15, random_state=0)


#KNN
classifier= KNeighborsClassifier(n_neighbors=8 , metric='minkowski', p=2 )
classifier.fit(x_train, y_train) 

#predicting the test set result
y_pred= classifier.predict(x_test)

#Creating the Confusion matrix 
cm= confusion_matrix(y_test, y_pred)

score = a_s(y_test, y_pred)

# pickling the model
import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()