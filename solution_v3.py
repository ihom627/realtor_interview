#use of XGBoost built in categorical variable support

#Solution using pandas dataframe
#cleanup input data to get improved results

#data in penguins_modified.csv

#STEP1) run at command line to setup
#pip3 install xgboost
#pip3 install scikit-learn
#pip3 install matplotlib
#pip3 install seaborn 
#pip3 install pandas 
#pip3 install numpy 
#pip3 install category_encoders
#brew install libomp

#STEP2) load packages
#python3
import pandas as pd
from sklearn.preprocessing import LabelEncoder          
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from category_encoders import OneHotEncoder




#STEP3 read into pandas dataframe
dataset = pd.read_csv('/Users/ivan.hom/projects/penguins/penguins_modified.csv')

dataset

dataset.dtypes


#convert object into categorical (or string)
dataset['species_short'] = dataset.species_short.astype('category')
dataset['island'] = dataset.island.astype('category')
dataset['sex'] = dataset.sex.astype('category')

dataset.dtypes

#has NaN values
dataset['sex'].value_counts()
dataset['species_short'].value_counts()
dataset['island'].value_counts()


dataset.head

#replace rows with values '.' and 'EMALE'  with NaN
dataset['sex'].describe()
dataset['sex'].value_counts()
dataset['sex'] = dataset['sex'].cat.remove_categories(['.'])
dataset['sex'] = dataset['sex'].cat.remove_categories(['EMALE'])

#Drop rows with any NaN values
dataset.dropna(inplace=True)



############### START SKIP #####################
#transform categorical to int64 labels
le = LabelEncoder()
dataset_v3['species_short'] = le.fit_transform(dataset_v3['species_short'])
dataset_v3['island'] = le.fit_transform(dataset_v3['island'])
dataset_v3['sex'] = le.fit_transform(dataset_v3['sex'])

dataset_v3.dtypes
################# END SKIP ####################



#split into train and test
X = dataset.drop('species_short', axis=1)
y = dataset['species_short']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#################### TRY APPLYING ONE HOT ENCODING ###################

X_train
X_train_v2 = pd.get_dummies(X_train, columns=['sex'])
X_train_v3 = pd.get_dummies(X_train_v2, columns=['island'])

X_test
X_test_v2 = pd.get_dummies(X_test, columns=['sex'])
X_test_v3 = pd.get_dummies(X_test_v2, columns=['island'])


y_train
y_train_v2 = pd.get_dummies(y_train, columns=['species_short'])

y_test
y_test_v2 = pd.get_dummies(y_test, columns=['species_short'])


model.fit(X_train, y_train)
model.fit(X_train, y_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/ivan.hom/Library/Python/3.9/lib/python/site-packages/xgboost/core.py", line 726, in inner_f
    return func(**kwargs)
  File "/Users/ivan.hom/Library/Python/3.9/lib/python/site-packages/xgboost/sklearn.py", line 1491, in fit
    raise ValueError(
ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2], got ['Adelie' 'Chinstrap' 'Gentoo']

############## END ONE HOT ENCODING #################################3


#STEP4 #train model

##### APPLY XGBOOST CATEGORICAL VARIABLES ##############
#model = XGBClassifier()
model = XGBClassifier(enable_categorical=True, tree_method='hist')

model.fit(X_train, y_train)

######### ERROR in fitting model ############
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/ivan.hom/Library/Python/3.9/lib/python/site-packages/xgboost/core.py", line 726, in inner_f
    return func(**kwargs)
  File "/Users/ivan.hom/Library/Python/3.9/lib/python/site-packages/xgboost/sklearn.py", line 1491, in fit
    raise ValueError(
ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2], got ['Adelie' 'Chinstrap' 'Gentoo']

######### END ERROR #########################



y_pred = model.predict(X_test)


#STEP5 show accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#>Accuracy: 0.9605263157894737 



#Try doing one-hot encoding instead to see if there is difference

from sklearn.preprocessing import OneHotEncoder

# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

# passing species_short column (label encoded values of species_short)
species_short_enc_df = pd.DataFrame(enc.fit_transform(dataset_v3[['species_short']]).toarray())

# merge with main df bridge_df on key values
dataset_v4 = dataset_v3.join(species_short_enc_df)
dataset_v4






