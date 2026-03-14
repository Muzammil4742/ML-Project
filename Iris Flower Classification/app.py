import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Dataset Read  
df = pd.read_csv("IRIS.csv")
print("Dataset loaded successfully!")
print(df.head())

# Drop missing values
df.dropna(inplace=True)

# Correlation Heatmap
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_cols.corr()




# Pie chart of class distribution
class_counts = df["species"].value_counts()

plt.figure(figsize=(8,8))
plt.pie(class_counts, labels=class_counts.index, autopct="%1.2f%%", startangle=90)
plt.title("Class Distribution in Iris Dataset")
plt.axis("equal")
plt.show()



# Select only numeric features
X = df.drop("species", axis=1)
y = df["species"]

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Normalization 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Selection
print("\nSelect Classification model from this:")
print("K Nearest Neighbour")
print("Decision Tree")
print("Random Forest")
classification_model = input("Enter Classification model: ").strip()

# Train Model
if classification_model == "K Nearest Neighbour":
    model = KNeighborsClassifier()
    model.fit(X_train_scaled, y_train)

elif classification_model == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

elif classification_model == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

else:
    print("Enter correct classification model")
    exit()

# Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')

print(f"\nModel trained: {classification_model}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")

# Predict on New User Input 
print(f"\nEnter feature values in the same order as dataset features")
features = input("Enter feature values (comma-separated): ").split(',')
features = [float(f.strip()) for f in features]

feature_columns = X.columns
features_df = pd.DataFrame([features], columns=feature_columns)

features_array_scaled = scaler.transform(features_df)

# Predict
y_pred_new = model.predict(features_array_scaled)
print(f"Estimated classification for the input features {features} is: {y_pred_new[0]}")
