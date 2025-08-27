import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ---------------- Constants ----------------
DATA_PATH = "data/raw/power_data.csv"
MODEL_PATH = "models/predictive_pipeline.pkl"
OUTPUT_DIR = "plots"

# ---------------- Create output folder ----------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Load dataset ----------------
df = pd.read_csv(DATA_PATH)
df = df.dropna()

# ---------------- Feature Engineering ----------------
df['Temp_Ratio'] = df['Air temperature [K]'] / df['Process temperature [K]']
df['Torque_per_RPM'] = df['Torque [Nm]'] / df['Rotational speed [rpm]']

# ---------------- 1. Failure Distribution ----------------
plt.figure(figsize=(6,4))
sns.countplot(x='Machine failure', data=df)
plt.title("Machine Failure Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "failure_distribution.png"))
plt.close()

# ---------------- 2. Numeric Feature Distributions (Individual) ----------------
numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 
                'Tool wear [min]', 'Temp_Ratio', 'Torque_per_RPM']

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=20, kde=True, color='skyblue')
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{col}_distribution.png"))
    plt.close()

# ---------------- 3. Correlation Heatmap ----------------
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# ---------------- 4. Scatter Plot Example ----------------
plt.figure(figsize=(6,4))
sns.scatterplot(x='Tool wear [min]', y='Torque [Nm]', hue='Machine failure', data=df)
plt.title("Tool Wear vs Torque")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_tool_torque.png"))
plt.close()

# ---------------- 5. Pairplot ----------------
sns.pairplot(df, hue='Machine failure', vars=['Air temperature [K]', 'Torque [Nm]', 'Tool wear [min]'])
plt.savefig(os.path.join(OUTPUT_DIR, "pairplot.png"))
plt.close()

# ---------------- 6. Feature Importance ----------------
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']

    # Get column names after preprocessing
    cat_cols = preprocessor.transformers_[1][1].get_feature_names_out()
    num_cols = preprocessor.transformers_[0][2]
    columns = list(num_cols) + list(cat_cols)

    importances = pd.Series(model.feature_importances_, index=columns)

    # Aggregate categorical features
    categorical_features = ['Product ID', 'Type']  # original categorical columns
    num_importances = importances[num_cols]
    cat_importances = {}
    for col in categorical_features:
        cat_cols_group = [c for c in importances.index if c.startswith(col)]
        cat_importances[col] = importances[cat_cols_group].sum()

    # Combine numeric + aggregated categorical importances
    all_importances = pd.concat([num_importances, pd.Series(cat_importances)])
    all_importances.sort_values(ascending=False).plot(kind='bar', figsize=(12,5))
    plt.title("Feature Importance (Aggregated)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    print("✅ Feature importance plotted")
else:
    print("⚠️ Pipeline not found. Run train.py first to generate feature importance.")

print(f"\n✅ All plots saved in '{OUTPUT_DIR}' folder")
