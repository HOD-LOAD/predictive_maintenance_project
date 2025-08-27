import pandas as pd
import joblib
import os
import argparse

PIPELINE_PATH = "models/predictive_pipeline.pkl"

def main(input_file, output_file=None):
    # --- Load pipeline ---
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError("Pipeline not found. Run train.py first.")
    pipeline = joblib.load(PIPELINE_PATH)
    
    # --- Load new data ---
    df = pd.read_csv(input_file)
    
    # --- Feature engineering (same as training) ---
    df['Temp_Ratio'] = df['Air temperature [K]'] / df['Process temperature [K]']
    df['Torque_per_RPM'] = df['Torque [Nm]'] / df['Rotational speed [rpm]']
    
    # --- Predict ---
    predictions = pipeline.predict(df)
    df['Predicted_Failure'] = predictions
    
    # --- Optional: save predictions ---
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"✅ Predictions saved to {output_file}")
    else:
        print("\nPredictions:\n", df[['Predicted_Failure']].head())
    
    # --- Quick summary ---
    counts = df['Predicted_Failure'].value_counts()
    print("\nPrediction summary:")
    for label, count in counts.items():
        status = "Failure" if label == 1 else "No Failure"
        print(f"{status}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict machine failure from new data")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", help="Path to save predictions CSV file (optional)")
    args = parser.parse_args()
    
    main(args.input, args.output)
