# Cancer prediction dataset: https://www.kaggle.com/datasets/erdemtaha/cancer-data/data?select=Cancer_Data.csv

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Data importing and preprocessing
# Normalizes the features, and converts output to binary int
# To prevent overfit, fewer features are selected  (just 3 columns out of 31)
dataset = pd.read_csv("./cancer_data.csv")
x = pd.DataFrame(MinMaxScaler().fit_transform(dataset.iloc[:, 2:5]))        # Feature selection (full: 2:32)
y = dataset.iloc[:, 1].replace({'B': 0, 'M': 1}).infer_objects(copy=False)  # Output [1/0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Model fitting and prediction
# SAG: Stochastic Average Gradient descent solver
logistic_model = LogisticRegression(solver='sag')
logistic_model.fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)

# Performance report
accuracy = accuracy_score(y_test, y_pred) * 100
confusion_mat = confusion_matrix(y_test, y_pred)
print("\n[MODEL ACCURACY]:", accuracy, "%")
print("\n[CONFUSION MATRIX]:\n", confusion_mat, "\n")