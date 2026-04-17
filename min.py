import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# 1. Load data
# ----------------------------
data = pd.read_csv(
    "ratings.csv",
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)

data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")

# ----------------------------
# 2. Feature engineering
# ----------------------------
user_features = data.groupby("userId").agg({
    "rating": ["count", "mean"],
    "timestamp": "max"
})

user_features.columns = ["rating_count", "rating_mean", "last_activity"]
user_features = user_features.reset_index()

# ----------------------------
# 3. Churn label (GROUND TRUTH)
# ----------------------------
reference_date = data["timestamp"].max()

user_features["days_inactive"] = (
    reference_date - user_features["last_activity"]
).dt.days

user_features["churn"] = (user_features["days_inactive"] > 60).astype(int)

# ----------------------------
# 4. FEATURE ENGINEERING (IMPORTANT FIX)
#    DO NOT use days_inactive directly
# ----------------------------

user_features["activity_rate"] = (
    user_features["rating_count"] / (user_features["days_inactive"] + 1)
)

user_features["is_high_rater"] = (user_features["rating_mean"] > 3.5).astype(int)

# ----------------------------
# 5. MODEL INPUT (NO LEAKAGE)
# ----------------------------
X = user_features[[
    "rating_count",
    "rating_mean",
    "activity_rate",
    "is_high_rater"
]]

y = user_features["churn"]

# ----------------------------
# 6. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 7. Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# 8. Prediction + Accuracy
# ----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

# ----------------------------
# 9. TABLE OUTPUT
# ----------------------------
print("\n--- USER FEATURES SAMPLE ---")
print(user_features.head(10))

results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred

print("\n--- PREDICTIONS SAMPLE ---")
print(results.head(10))