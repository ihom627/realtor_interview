#Solution using pandas dataframe

#data in penguins_modified.csv

#STEP1) run at command line to setup
#pip3 install xgboost
#pip3 install scikit-learn
#pip3 install matplotlib
#pip3 install seaborn 
#pip3 install pandas 
#pip3 install numpy 
#brew install libomp

#STEP2) load packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder          
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



#STEP3 read into pandas dataframe
dataset = pd.read_csv('/Users/ivan.hom/projects/penguins/penguins_modified.csv')

dataset.dtypes

#convert object into categorical (or string)
dataset['species_short'] = dataset.species_short.astype('category')
dataset['island'] = dataset.island.astype('category')
dataset['sex'] = dataset.sex.astype('category')

dataset.dtypes

#transform categorical to int64 labels
le = LabelEncoder()
dataset['species_short'] = le.fit_transform(dataset['species_short'])
dataset['island'] = le.fit_transform(dataset['island'])
dataset['sex'] = le.fit_transform(dataset['sex'])

dataset.dtypes


#split into train and test
X = dataset.drop('species_short', axis=1)
y = dataset['species_short']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#STEP4 #train model
model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


#STEP5 show accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#>Accuracy: 0.9493670886075949

