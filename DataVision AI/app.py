from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "datasets"
GRAPH_FOLDER = "static/graphs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return send_from_directory("template", "index.html")


# 1️⃣ Upload
@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file selected"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    return jsonify({"message": "File uploaded successfully", "path": path})


# 2️⃣ Clean
@app.route("/clean", methods=["POST"])
def clean_dataset():
    data = request.json
    df = pd.read_csv(data["path"])

    df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True, errors="ignore")

    save_path = data["path"].replace(".csv", "_cleaned.csv")
    df.to_csv(save_path, index=False)

    return jsonify({"path": save_path})


# 3️⃣ Normalize
@app.route("/normalize", methods=["POST"])
def normalize_dataset():
    data = request.json
    df = pd.read_csv(data["path"])

    numeric = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    save_path = data["path"].replace(".csv", "_normalized.csv")
    df.to_csv(save_path, index=False)

    return jsonify({"path": save_path})



@app.route("/dtypes", methods=["POST"])
def show_dtypes():
    data = request.json
    df = pd.read_csv(data["path"])
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    return jsonify({"data_types": dtypes})



# 4️⃣ Correlation (col1 vs col2)
@app.route("/correlation", methods=["POST"])
def correlation_two_cols():
    data = request.json
    df = pd.read_csv(data["path"])

    col1 = data.get("col1")
    col2 = data.get("col2")

    if col1 not in df.columns or col2 not in df.columns:
        return jsonify({"error": "Invalid column names"}), 400

    corr = df[col1].corr(df[col2])

    return jsonify({"correlation": float(corr)})









# 5️⃣ Train Model + Feature → Target Correlation
@app.route("/train_model", methods=["POST"])
def train_model():
    data = request.json

    df = pd.read_csv(data["path"])
    features = data["features"]
    target = data["target"]
    model_type = data["model_type"]
    eval_type = data["eval_type"]

    df = df[features + [target]].dropna()
    X = df[features].astype(float)
    y = df[target].astype(float)

    # --- Feature correlations ---
    feature_corr = {}
    for f in features:
        c = df[f].corr(df[target])
        feature_corr[f] = {"correlation": round(c, 3)}

    # Pick model
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=0.1)
    elif model_type == "polynomial":
        model = Pipeline([
            ('scale', StandardScaler()),
            ('poly', PolynomialFeatures(include_bias=False)),
            ('model', LinearRegression())
        ])
    else:
        return jsonify({"error": "Unsupported model"}), 400

    # --- Evaluation ---
    mse = None
    r2 = None

    if eval_type == "split":
        test_size = float(data.get("test_size", 0.2))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

    elif eval_type == "kfold":
        k = int(data.get("k", 5))
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        mse_list = []
        r2_list = []

        for train_i, test_i in kf.split(X):
            model.fit(X.iloc[train_i], y.iloc[train_i])
            preds = model.predict(X.iloc[test_i])
            mse_list.append(mean_squared_error(y.iloc[test_i], preds))
            r2_list.append(r2_score(y.iloc[test_i], preds))

        mse = sum(mse_list) / len(mse_list)
        r2 = sum(r2_list) / len(r2_list)



    return jsonify({
        "feature_correlation": feature_corr,
        "evaluation": {
            "mse": round(mse, 4) if mse is not None else None,
            "r2": round(r2, 4) if r2 is not None else None
        }
    })



@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.read_csv(data["path"])
    features = data["features"]
    target = data["target"]
    model_type = data["model_type"]
    predict_values = data["predict_values"]

    X = df[features].dropna().astype(float)
    y = df[target].dropna().astype(float)

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=0.1)
    elif model_type == "polynomial":
        model = Pipeline([
            ('scale', StandardScaler()),
            ('poly', PolynomialFeatures(include_bias=False)),
            ('model', LinearRegression())
        ])
    else:
        return jsonify({"error": "Unsupported model"}), 400

    model.fit(X, y)
    prediction = model.predict([predict_values])[0]

    # Optional evaluation on full data
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    return jsonify({
        "prediction": float(prediction),
        "evaluation": {"mse": round(mse,4), "r2": round(r2,4)}
    })


# 6️⃣ Graph
@app.route("/graph", methods=["POST"])
def generate_graph():
    data = request.json
    df = pd.read_csv(data["path"])
    x_col = data.get("x")
    y_col = data.get("y")
    graph_type = data.get("graph_type")

    plt.figure(figsize=(8,6))

    try:
        if graph_type == "scatter":
            sns.scatterplot(data=df, x=x_col, y=y_col)
        elif graph_type == "line":
            sns.lineplot(data=df, x=x_col, y=y_col)
        elif graph_type == "bar":
            sns.barplot(data=df, x=x_col, y=y_col)
        elif graph_type == "regression":
            sns.regplot(data=df, x=x_col, y=y_col)
        elif graph_type == "heatmap":
            numeric_df = df.select_dtypes(include=['float64','int64'])
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    plot_path = "/static/graphs/graph_{}_{}.png".format(x_col, y_col)
    plt.savefig("." + plot_path)  # Save to project folder
    plt.close()

    return jsonify({"graph_path": plot_path})



# 7️⃣ Serve Graphs
@app.route("/static/graphs/<path:filename>")
def serve_graph(filename):
    return send_from_directory("static/graphs", filename)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
