import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Constants
DATA_PATH = "data/raw/power_data.csv"
MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "predictive_pipeline.pkl")
TARGET_COL = "Machine failure"

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    
    # Feature engineering
    df['Temp_Ratio'] = df['Air temperature [K]'] / df['Process temperature [K]']
    df['Torque_per_RPM'] = df['Torque [Nm]'] / df['Rotational speed [rpm]']
    
    # Features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # Model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1]))
    
    # Save pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"\n✅ Pipeline saved at {PIPELINE_PATH}")

if __name__ == "__main__":
    main()
