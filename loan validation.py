import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
loan_dataset = pd.read_csv(r'C:\Users\thenn\OneDrive\Desktop\loan_approval_dataset.csv')

# Drop ID column
loan_dataset = loan_dataset.drop(columns='loan_id', axis=1)

# Handle missing values (drop or fill)
loan_dataset = loan_dataset.dropna()  # or use fillna()

# Label Encoding
loan_dataset.replace({
    ' loan_status': {' Approved': 1, " Rejected": 0},
    ' education': {' Graduate': 1, ' Not Graduate': 0},
    ' self_employed': {' No': 0, ' Yes': 1}
}, inplace=True)

# Features and target
x = loan_dataset.drop(columns=' loan_status', axis=1)
y = loan_dataset[' loan_status']

# Feature scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Train-test split (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train the SVM model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Accuracy on train and test
x_train_pred = classifier.predict(x_train)
x_test_pred = classifier.predict(x_test)

print(f"Training Accuracy: {accuracy_score(y_train, x_train_pred)}")
print(f"Testing Accuracy: {accuracy_score(y_test, x_test_pred)}")

# Predicting for one instance
input_data = x_test[0].reshape(1, -1)  # reshape to 2D
prediction = classifier.predict(input_data)

print("Actual:", y_test.iloc[0])
print("Predicted:", "Approved" if prediction[0] == 1 else "Rejected")
