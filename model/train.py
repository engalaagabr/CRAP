# =====================================================
# Credit Risk Multiclass Model - (FINAL)
# =====================================================

import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import xgboost as xg

# -----------------------------------------------------
# 1. Load Data
# -----------------------------------------------------
data = pd.read_csv("processed_data.csv")
# -----------------------------------------------------
# 3. Split Features / Target
# -----------------------------------------------------
X = data.drop('Credit_Score', axis=1)
y = data['Credit_Score']

# -----------------------------------------------------
# 4. Column Groups (MATCH YOUR DATA EXACTLY)
# -----------------------------------------------------
categorical_cols = [
    'Credit_Mix',
    'Payment_of_Min_Amount'
]

numerical_cols = [
    c for c in X.columns if c not in categorical_cols
]

# -----------------------------------------------------
# 5. Preprocessing
# -----------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# -----------------------------------------------------
# 6. XGBoost Model
# -----------------------------------------------------
xgb_model = xg.XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# -----------------------------------------------------
# 7. FULL PIPELINE
# -----------------------------------------------------
pipeline = ImbPipeline(steps=[
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", xgb_model)
])

# -----------------------------------------------------
# 8. Train on FULL DATASET
# -----------------------------------------------------
print("🚀 Training XGBoost model on full dataset...")
pipeline.fit(X, y)

# -----------------------------------------------------
# 9. Save Model
# -----------------------------------------------------
joblib.dump(pipeline, "model.pkl")

print("✅ Model saved as model.pkl")
