# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Import necessary libraries and load the data file `encoded_car_data (1).csv`.
3. Select the input features (‘enginesize,’ ‘horsepower,’ ‘citympg,’ ‘high
4. Split the dataset into training set and test set in 80:20 proportions.
5. Create a linear regression pipeline using standard scalers.
6. Train the Linear Regression model and predict the test data.
7. Create a Polynomial Regression pipeline, specifying a degree of 2 and scaling.
8. Train the Polynomial model and predict the test data.
9. Calculate MSE, MAE, and R² score to compare the performance.
10. Plot Actual vs Predicted values for both models and stop the program.



## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import matplotlib.pyplot as plt
df = pd.read_csv('encoded_car_data (1).csv')
print(df.head())
X = df[['enginesize', 'horsepower','citympg', 'highwaympg']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
lr.fit(X_train, y_train)
y_pred_linear = lr.predict(X_test)
poly_model =Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
print('Name:Carlton Maximus A')
print('reg: 212225040052')
print('Linear Regeression')
print('MSE=',mean_squared_error(y_test,y_pred_linear))
print('MAE=',mean_absolute_error(y_test,y_pred_linear))
r2score=r2_score(y_test,y_pred_linear)
print('R2 Score=',r2score)
print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test,y_pred_poly):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_poly):.2f}")
plt.figure(figsize=(10, 5))
plt.scatter(y_test,y_pred_linear, label='Linear', alpha=0.6)
plt.scatter(y_test,y_pred_poly, label='Polynomial(degree=2)', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Prefect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomail Regression")
plt.legend()
plt.show()
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: Carlton Maximus A
RegisterNumber: 212225040052
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="836" height="481" alt="image" src="https://github.com/user-attachments/assets/c4951fcd-6174-408b-9620-249891e2ea33" />
<img width="471" height="229" alt="image" src="https://github.com/user-attachments/assets/e63f831e-0606-4527-b0a9-a2d6d1f48587" />
<img width="1167" height="604" alt="image" src="https://github.com/user-attachments/assets/6e3ab7bb-aa39-417e-b82e-4aac0bc0f24c" />


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
