import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# 1. Load dataset
df = pd.read_csv("possum (1).csv")

# 2. Handle missing values
df = df.dropna()

# 3. Define target & features
y = df["age"]   # target column
X = df.drop(columns=["age"])

# Convert categorical features
X = pd.get_dummies(X, drop_first=True)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train XGBoost Regressor
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("✅ R2 Score:", r2_score(y_test, y_pred))
print("✅ RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 7. Save model + feature names
with open("xgb_age_model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("🎉 Model saved as xgb_age_model.pkl")
