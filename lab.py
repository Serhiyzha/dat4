import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", 
           "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", 
           "length", "width", "height", "curb-weight", "engine-type", 
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", 
           "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df = pd.read_csv(url, header=None, names=headers)
df.replace("?", np.nan, inplace=True)

numeric_cols = ["engine-size", "price", "horsepower", "curb-weight", "highway-mpg"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col])

df.dropna(subset=numeric_cols, inplace=True)

X_sample = df[["horsepower"]].iloc[:25]
y_sample = df["price"].iloc[:25]

x_train, x_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

overfit_model = make_pipeline(PolynomialFeatures(degree=10), StandardScaler(), LinearRegression())
overfit_model.fit(x_train, y_train)

print("Overfitting:")
print(f"R^2 (train): {r2_score(y_train, overfit_model.predict(x_train)):.4f}")
print(f"R^2 (test):  {r2_score(y_test, overfit_model.predict(x_test)):.4f}")

x_plot = np.linspace(X_sample.min(), X_sample.max(), 200).reshape(-1, 1)
x_plot_df = pd.DataFrame(x_plot, columns=["horsepower"])

plt.figure(figsize=(8, 4))
plt.scatter(x_train, y_train, color="blue", label="Train дані")
plt.scatter(x_test, y_test, color="green", label="Test дані")
plt.plot(x_plot, overfit_model.predict(x_plot_df), color="red", label="Поліном 10-го ступеня")
plt.ylim(0, df["price"].max() * 1.2)
plt.title("Overfitting")
plt.xlabel("horsepower")
plt.ylabel("price")
plt.legend()
plt.grid(True)
plt.show()

underfit_model = make_pipeline(PolynomialFeatures(degree=0), LinearRegression())
underfit_model.fit(x_train, y_train)

print("\nUnderfitting:")
print(f"R^2 (train): {r2_score(y_train, underfit_model.predict(x_train)):.4f}")
print(f"R^2 (test):  {r2_score(y_test, underfit_model.predict(x_test)):.4f}")

plt.figure(figsize=(8, 4))
plt.scatter(x_train, y_train, color="blue", label="Train дані")
plt.scatter(x_test, y_test, color="green", label="Test дані")
plt.plot(x_plot, underfit_model.predict(x_plot_df), color="orange", label="Константна модель (degree=0)")
plt.title("Underfitting")
plt.xlabel("horsepower")
plt.ylabel("price")
plt.legend()
plt.grid(True)
plt.show()

X_features = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
Y_target = df["price"]

parameters = [{"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}]
ridge = Ridge()

grid = GridSearchCV(ridge, parameters, cv=4)
grid.fit(X_features, Y_target)

print("\nGridSearch для Ridge Regression:")
print(f"Найкращий гіперпараметр: {grid.best_params_}")
print(f"Найкращий результат (R^2): {grid.best_score_:.4f}")