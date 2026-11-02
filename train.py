"""Train, compare and save the best house price model."""
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

for d in ("models", "screenshots"):
    os.makedirs(d, exist_ok=True)

# ---------- 1. Load ----------
data = pd.read_csv("data/house_data.csv")
print(data.head(), "\n")

# ---------- 2. Explore ----------
data.info()
print("\nMissing values:\n", data.isnull().sum())
print("\n", data.describe())
data = data.dropna().drop_duplicates()

# ---------- 3. Visualize ----------
plt.figure(figsize=(6, 4))
plt.scatter(data["Area"], data["Price"], alpha=0.4)
plt.xlabel("Area (sq ft)"); plt.ylabel("Price (Lakh)")
plt.title("Area vs Price"); plt.tight_layout()
plt.savefig("screenshots/area_vs_price.png"); plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(data.select_dtypes("number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig("screenshots/correlation.png"); plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(x="Location", y="Price", data=data)
plt.xticks(rotation=30); plt.tight_layout()
plt.savefig("screenshots/price_by_location.png"); plt.close()

# ---------- 4. Preprocess ----------
FEATURES = ["Area", "Bedrooms", "Bathrooms", "Location", "Age", "Parking"]
X, y = data[FEATURES], data["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer(
    [("loc", OneHotEncoder(handle_unknown="ignore"), ["Location"])],
    remainder="passthrough")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42,
                                           n_jobs=-1),
}

# ---------- 5. Train + Evaluate ----------
results, fitted = {}, {}
for name, reg in models.items():
    pipe = Pipeline([("pre", pre), ("model", reg)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    results[name] = {"MAE": round(mean_absolute_error(y_test, pred), 3),
                     "MSE": round(mse, 3),
                     "RMSE": round(float(np.sqrt(mse)), 3),
                     "R2": round(r2_score(y_test, pred), 4)}
    fitted[name] = pipe
    print(f"\n{name}: {results[name]}")

best = max(results, key=lambda k: results[k]["R2"])
print(f"\nBest model: {best}")

# Actual vs predicted plot for best model
pred = fitted[best].predict(X_test)
plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred, alpha=0.4)
lims = [y.min(), y.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"{best}: Actual vs Predicted")
plt.tight_layout(); plt.savefig("screenshots/actual_vs_predicted.png"); plt.close()

# ---------- 6. Save ----------
joblib.dump(fitted["Linear Regression"], "models/linear_model.pkl")
joblib.dump(fitted["Random Forest"], "models/rf_model.pkl")
joblib.dump(fitted[best], "models/best_model.pkl")
meta = {"best_model": best, "results": results,
        "locations": sorted(data["Location"].unique().tolist()),
        "features": FEATURES}
with open("models/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved models to models/")
