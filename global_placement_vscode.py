"""
VS Code / Local Python friendly training script for global_placement.csv

How to use:
1. Put this file in the same folder as your CSV file
2. Rename/update CSV_FILE if needed
3. Run:
   python global_placement_vscode.py

Required packages:
pip install pandas numpy seaborn matplotlib scikit-learn xgboost joblib

What this script does:
- Classification: predicts placement_status
- Regression: predicts salary only for placed students
- Saves best models locally as .pkl files

Author note:
This version does NOT use any Google Colab-only code.
"""

# ============================================================
# STEP 1: Import libraries
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Classification models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Classification metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost models
from xgboost import XGBClassifier, XGBRegressor


# ============================================================
# STEP 2: Configuration
# ============================================================
# Edit these values only if your file/column names are different.
# ============================================================

CSV_FILE = "global_placement.csv"
CLASSIFICATION_TARGET = "placement_status"
REGRESSION_TARGET = "salary"
DROP_COLUMNS = ["student_id", "name"]

# Plot settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)


# ============================================================
# STEP 3: Utility functions
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_roc_auc(y_true, y_prob):
    """Return ROC-AUC safely. If it fails, return NaN."""
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def plot_tree_feature_importance(trained_pipeline, title, top_n=20, save_path=None):
    """
    Plot feature importance for tree-based models.
    Because one-hot encoding expands features, we use the transformed names.
    """
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    model = trained_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        print(f"Skipping feature importance for {title} (model has no feature_importances_)")
        return

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def save_confusion_matrix(cm, labels, title, save_path=None):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


# ============================================================
# STEP 4: Load dataset
# ============================================================

print_section("STEP 4: Load dataset")

csv_path = Path(CSV_FILE)

if not csv_path.exists():
    raise FileNotFoundError(
        f"Could not find '{CSV_FILE}'. Put the CSV in the same folder as this script "
        f"or change CSV_FILE at the top of the script."
    )

df = pd.read_csv(csv_path)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# STEP 5: Basic inspection
# ============================================================

print_section("STEP 5: Basic inspection")
print("Columns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# STEP 6: Basic target cleanup
# ============================================================
# This standardizes common text labels like:
# Placed / Not Placed / Yes / No / Selected / Not Selected
# ============================================================

print_section("STEP 6: Clean placement target labels")

if CLASSIFICATION_TARGET not in df.columns:
    raise ValueError(f"Column '{CLASSIFICATION_TARGET}' not found in CSV.")

if REGRESSION_TARGET not in df.columns:
    raise ValueError(f"Column '{REGRESSION_TARGET}' not found in CSV.")

if df[CLASSIFICATION_TARGET].dtype == "object":
    df[CLASSIFICATION_TARGET] = (
        df[CLASSIFICATION_TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    label_map = {
        "placed": "placed",
        "not placed": "not placed",
        "not_placed": "not placed",
        "yes": "placed",
        "no": "not placed",
        "selected": "placed",
        "not selected": "not placed",
    }
    df[CLASSIFICATION_TARGET] = df[CLASSIFICATION_TARGET].replace(label_map)

print("Unique placement_status values:")
print(df[CLASSIFICATION_TARGET].unique())


# ============================================================
# STEP 7: Classification task
# ============================================================
# Predict placement_status
# ============================================================

print_section("STEP 7: Classification task")

drop_cols_for_classification = DROP_COLUMNS + [CLASSIFICATION_TARGET]
if REGRESSION_TARGET in df.columns:
    drop_cols_for_classification.append(REGRESSION_TARGET)

drop_cols_for_classification = [c for c in drop_cols_for_classification if c in df.columns]

X_class = df.drop(columns=drop_cols_for_classification).copy()
y_class_raw = df[CLASSIFICATION_TARGET].copy()

# Encode target labels into numbers
label_encoder = LabelEncoder()
y_class = label_encoder.fit_transform(y_class_raw)

print("Encoded class mapping:")
for idx, cls_name in enumerate(label_encoder.classes_):
    print(f"{cls_name} -> {idx}")

# Separate numeric and categorical features
numeric_features = X_class.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_class.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# Preprocessing pipelines
numeric_transformer_scaled = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_transformer_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

scaled_preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer_scaled, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

tree_preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer_tree, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Train-test split
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class,
)

print("\nTrain shape:", X_train_class.shape)
print("Test shape :", X_test_class.shape)

# Classification models
classification_models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", scaled_preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),

    "Random Forest": Pipeline(steps=[
        ("preprocessor", tree_preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "XGBoost": Pipeline(steps=[
        ("preprocessor", tree_preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        ))
    ]),

    "SVC": Pipeline(steps=[
        ("preprocessor", scaled_preprocessor),
        ("model", SVC(probability=True, kernel="rbf", random_state=42))
    ]),

    "MLP": Pipeline(steps=[
        ("preprocessor", scaled_preprocessor),
        ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42
        ))
    ]),
}

classification_results = []
trained_classification_models = {}

for model_name, pipeline in classification_models.items():
    print(f"\nTraining classification model: {model_name}")
    pipeline.fit(X_train_class, y_train_class)
    trained_classification_models[model_name] = pipeline

    y_pred = pipeline.predict(X_test_class)

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test_class)[:, 1]
    else:
        y_prob = None

    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test_class, y_pred),
        "Precision": precision_score(y_test_class, y_pred, zero_division=0),
        "Recall": recall_score(y_test_class, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test_class, y_pred, zero_division=0),
        "ROC-AUC": safe_roc_auc(y_test_class, y_prob) if y_prob is not None else np.nan,
    }

    classification_results.append(result)

classification_results_df = (
    pd.DataFrame(classification_results)
    .sort_values(by="F1 Score", ascending=False)
    .reset_index(drop=True)
)

print_section("Classification results")
print(classification_results_df)

# Best classification model
best_classification_model_name = classification_results_df.iloc[0]["Model"]
best_classification_model = trained_classification_models[best_classification_model_name]

print(f"\nBest classification model: {best_classification_model_name}")

y_pred_best = best_classification_model.predict(X_test_class)

print("\nDetailed classification report:")
print(classification_report(
    y_test_class,
    y_pred_best,
    target_names=label_encoder.classes_
))

cm = confusion_matrix(y_test_class, y_pred_best)
save_confusion_matrix(
    cm,
    label_encoder.classes_,
    f"Confusion Matrix - {best_classification_model_name}",
    save_path="best_classification_confusion_matrix.png"
)

# Classification comparison plot
plot_df = classification_results_df.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(12, 6))
sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
plt.title("Classification Model Comparison")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("classification_model_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

# Feature importance
if "Random Forest" in trained_classification_models:
    plot_tree_feature_importance(
        trained_classification_models["Random Forest"],
        "Top Classification Features - Random Forest",
        save_path="rf_classification_feature_importance.png"
    )

if "XGBoost" in trained_classification_models:
    plot_tree_feature_importance(
        trained_classification_models["XGBoost"],
        "Top Classification Features - XGBoost",
        save_path="xgb_classification_feature_importance.png"
    )


# ============================================================
# STEP 8: Regression task
# ============================================================
# Predict salary only for placed students
# ============================================================

print_section("STEP 8: Regression task")

# Find the "Placed" class name from the label encoder
placed_class_name = None
for cls_name in label_encoder.classes_:
    if cls_name.lower() == "placed":
        placed_class_name = cls_name
        break

if placed_class_name is None:
    placed_class_name = "Placed"  # fallback

# Create mask for placed students using the actual class name
placed_mask = df[CLASSIFICATION_TARGET] == placed_class_name

df_reg = df.loc[placed_mask].copy()

print("Placed-only dataset shape:", df_reg.shape)

drop_cols_for_regression = DROP_COLUMNS + [CLASSIFICATION_TARGET, REGRESSION_TARGET]
drop_cols_for_regression = [c for c in drop_cols_for_regression if c in df_reg.columns]

X_reg = df_reg.drop(columns=drop_cols_for_regression).copy()
y_reg = df_reg[REGRESSION_TARGET].copy()

numeric_features_reg = X_reg.select_dtypes(include=[np.number]).columns.tolist()
categorical_features_reg = X_reg.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumeric regression features:")
print(numeric_features_reg)

print("\nCategorical regression features:")
print(categorical_features_reg)

# Preprocessing for regression
numeric_transformer_scaled_reg = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_transformer_tree_reg = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer_reg = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

scaled_preprocessor_reg = ColumnTransformer(transformers=[
    ("num", numeric_transformer_scaled_reg, numeric_features_reg),
    ("cat", categorical_transformer_reg, categorical_features_reg)
])

tree_preprocessor_reg = ColumnTransformer(transformers=[
    ("num", numeric_transformer_tree_reg, numeric_features_reg),
    ("cat", categorical_transformer_reg, categorical_features_reg)
])

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42,
)

print("\nRegression train shape:", X_train_reg.shape)
print("Regression test shape :", X_test_reg.shape)

# Regression models
regression_models = {
    "Linear Regression": Pipeline(steps=[
        ("preprocessor", scaled_preprocessor_reg),
        ("model", LinearRegression())
    ]),

    "Random Forest Regressor": Pipeline(steps=[
        ("preprocessor", tree_preprocessor_reg),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "XGBoost Regressor": Pipeline(steps=[
        ("preprocessor", tree_preprocessor_reg),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        ))
    ]),
}

regression_results = []
trained_regression_models = {}

for model_name, pipeline in regression_models.items():
    print(f"\nTraining regression model: {model_name}")
    pipeline.fit(X_train_reg, y_train_reg)
    trained_regression_models[model_name] = pipeline

    y_pred_reg = pipeline.predict(X_test_reg)

    result = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test_reg, y_pred_reg),
        "RMSE": np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
        "R2 Score": r2_score(y_test_reg, y_pred_reg),
    }

    regression_results.append(result)

regression_results_df = (
    pd.DataFrame(regression_results)
    .sort_values(by="R2 Score", ascending=False)
    .reset_index(drop=True)
)

print_section("Regression results")
print(regression_results_df)

# Best regression model
best_regression_model_name = regression_results_df.iloc[0]["Model"]
best_regression_model = trained_regression_models[best_regression_model_name]

print(f"\nBest regression model: {best_regression_model_name}")

y_pred_best_reg = best_regression_model.predict(X_test_reg)

# Actual vs predicted plot
plt.figure(figsize=(7, 7))
plt.scatter(y_test_reg, y_pred_best_reg, alpha=0.6)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title(f"Actual vs Predicted Salary - {best_regression_model_name}")

min_val = min(y_test_reg.min(), y_pred_best_reg.min())
max_val = max(y_test_reg.max(), y_pred_best_reg.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")
plt.tight_layout()
plt.savefig("salary_actual_vs_predicted.png", dpi=200, bbox_inches="tight")
plt.show()

# Regression feature importance
if "Random Forest Regressor" in trained_regression_models:
    plot_tree_feature_importance(
        trained_regression_models["Random Forest Regressor"],
        "Top Salary Features - Random Forest Regressor",
        save_path="rf_regression_feature_importance.png"
    )

if "XGBoost Regressor" in trained_regression_models:
    plot_tree_feature_importance(
        trained_regression_models["XGBoost Regressor"],
        "Top Salary Features - XGBoost Regressor",
        save_path="xgb_regression_feature_importance.png"
    )


# ============================================================
# STEP 9: Save best models locally
# ============================================================

print_section("STEP 9: Save models")

joblib.dump(best_classification_model, "best_classification_model.pkl")
joblib.dump(best_regression_model, "best_salary_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Saved files:")
print("- best_classification_model.pkl")
print("- best_salary_model.pkl")
print("- label_encoder.pkl")


# ============================================================
# STEP 10: Finish
# ============================================================

print_section("Finished")
print("All done. You can now inspect the saved .pkl files and generated plots.")
