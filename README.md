# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Data Collection and Preprocessing**
   Load the food nutrition dataset, handle missing values, encode categorical labels, and normalize the nutritional features using scaling techniques.

2. **Data Splitting**
   Split the dataset into training and testing sets using an appropriate ratio (e.g., 80% training and 20% testing).

3. **Model Training**
   Train a Logistic Regression model using the training dataset to learn the relationship between nutritional features and diabetic suitability.

4. **Model Evaluation and Prediction**
   Test the model on the testing dataset and evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix. Use the trained model to classify food items as suitable or unsuitable for diabetic patients.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('food_items (1).csv')
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
X_raw = df.iloc[:,:-1]
y_raw = df.iloc[:,-1:]
scaler= MinMaxScaler()
X= scaler.fit_transform(X_raw)
label_encoder = LabelEncoder()
y= label_encoder.fit_transform(y_raw.values.ravel())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify =y, random_state=123)

penalty='l2'

multi_class='multinomial'

solver= 'lbfgs'

max_iter=1000

l2_model = LogisticRegression(
    random_state=123,
    penalty=penalty,
    multi_class=multi_class, 
    solver=solver, 
    max_iter=max_iter
)
l2_model.fit(X_train, y_train)

y_pred= l2_model.predict(X_test)
print('Name:ROHITH S')
print('Reg. No:212225240121')
print("\nModel Evaluation:")
print("Accuracy:",accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)




```

## Output:
<img width="853" height="495" alt="image" src="https://github.com/user-attachments/assets/51eba136-3fb5-4618-9f5c-c7b240552d43" />
<img width="683" height="470" alt="image" src="https://github.com/user-attachments/assets/a63d3357-223a-47ee-b844-c43e9160642a" />
<img width="703" height="349" alt="image" src="https://github.com/user-attachments/assets/cc3bfcb9-1afe-4c36-b031-d6b4729b1c26" />


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
