import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
import numpy as np
# Dataset
df = pd.read_csv('penguins_lter.csv')


# Clean data

clean_Data = df.dropna(axis=1).dropna(axis=0)

# Use: Culmen Length (mm), Culmen Depth (mm), Flipper Length (mm) --> X
# Use: Species --> Y

X = df[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)']]
Y = df['Species']

cleaned_X = X.dropna()   

df_cleaned = df.dropna(subset=['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)'])

X = df_cleaned[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)']]
Y = df_cleaned['Species']

labelTranslator = LabelEncoder()

Y_new = labelTranslator.fit_transform(Y)

knn_Model = KNeighborsClassifier(n_neighbors=2)

X_train, x_test, Y_train, y_test = train_test_split(X, Y_new, test_size=0.3, random_state=42)

scalar = StandardScaler()
x_train_scalar = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(x_test)


knn_Model.fit(x_train_scalar, Y_train)

predict_Species = knn_Model.predict(X_test_scaled)

acc_Score = accuracy_score(y_test, predict_Species) * 100

print(f'The accuracy for this model {acc_Score:.2f} %')

invert = labelTranslator.inverse_transform(predict_Species)


test1 = pd.DataFrame([[50, 13, 175]], columns=['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)'])

test1_scaled = scalar.transform(test1)

pTestOne = knn_Model.predict(test1_scaled)

pTOtL = labelTranslator.inverse_transform(pTestOne)

print(f'Species of penguin 1: {pTOtL[0]}')

test2 = pd.DataFrame([[20, 76, 195]], columns=['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)'])

test2_scaled = scalar.transform(test2)

pTestTwo = knn_Model.predict(test2_scaled)

pTTtL = labelTranslator.inverse_transform(pTestTwo)

print(f'Species of penguin 2: {pTTtL[0]}')