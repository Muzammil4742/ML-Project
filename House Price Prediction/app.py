import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#  Dataset Read  
df = pd.read_csv("House selling data set.csv")
print("Dataset loaded successfully!")
print(df.head())

# Drop missing values
df.dropna(inplace=True)

# Correlation Heatmap
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(18,16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation of housing dataset')
plt.show()

# Check correlation between two columns 
print("\nAvailable columns:", df.columns.tolist())
col1 = input("Enter first column to check correlation: ").strip()
col2 = input("Enter second column to check correlation: ").strip()

if col1 in df.columns and col2 in df.columns:
    corr_coefficients = df[col1].corr(df[col2])
    print(f"Correlation Between {col1} and {col2} is: {corr_coefficients:.2f}")
else:
    print("Invalid column names entered!")

#  User selects features and target
print("\nAvailable columns:", df.columns.tolist())
feature_columns = input("Enter feature columns (comma-separated): ").split(',')
feature_columns = [col.strip() for col in feature_columns]

target_column = input("Enter target column: ").strip()

# Select only numeric features
X = df[feature_columns].select_dtypes(include=['int64', 'float64'])
y = df[target_column]

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Normalization 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Model Selectio
print("\nSelect Regression model from this:")
print("Linear")
print("Ridge")
print("Polynomial")
print("Random Forest")
regression_model = input("Enter regression model: ").strip()

#  Train Model
if regression_model == "Linear":
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

elif regression_model == "Ridge":
    model = Ridge()
    model.fit(X_train_scaled, y_train)

elif regression_model == "Polynomial":
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    X_train_scaled, X_test_scaled = X_train_poly, X_test_poly  

elif regression_model == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

else:
    print("Enter correct regression model")
    exit()

#Evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel trained: {regression_model}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

#  Predict on New User Input 
print(f"\nEnter {len(feature_columns)} feature values in the same order as {feature_columns}")
features = input("Enter feature values (comma-separated): ").split(',')
features = [float(f.strip()) for f in features]

# Convert input to DataFrame with same column names
features_df = pd.DataFrame([features], columns=feature_columns)

# Apply same scaling / polynomial transformation
features_array_scaled = scaler.transform(features_df)
if regression_model == "Polynomial":
    features_array_scaled = poly.transform(features_array_scaled)

# Predict
y_pred_new = model.predict(features_array_scaled)
print(f"Estimated value for the input features {features} is: {y_pred_new[0]:.2f}")
