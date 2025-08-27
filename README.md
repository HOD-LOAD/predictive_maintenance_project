# Predictive Maintenance of Industrial Machines

## Overview
This project implements a **Predictive Maintenance system** for industrial machines using machine learning. 
The goal is to predict machine failures in advance based on sensor data, helping reduce downtime and maintenance costs.

The project includes:
- Data preprocessing and **feature engineering**
- Training a machine learning model (**XGBoost**) for failure prediction
- Prediction on new datasets
- Exploratory Data Analysis (EDA) with **visualizations**
- Clean, presentation-ready plots and insights

---

## Features
- **Train model**: `train.py` trains the model with engineered features like `Temp_Ratio` and `Torque_per_RPM`.
- **Predict failures**: `predict.py` predicts machine failures for new datasets.
- **EDA & Visualizations**: `eda_visualization.py` generates:
  - Failure distribution plots  
  - Numeric feature distributions  
  - Correlation heatmap  
  - Scatter plots and pairplots  
  - Feature importance (aggregated categorical + numeric features)

---

## Project Structure
predictive_maintenance/
│
├─ data/raw/ # Raw sensor dataset
├─ models/ # Trained ML model pipeline
├─ plots/ # Generated plots from EDA
├─ src/
│ ├─ train.py
│ ├─ predict.py
│ └─ eda_visualization.py
├─ .gitignore
└─ README.md


---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/HOD-LOAD/predictive_maintenance_project.git

python -m venv .venv
source .venv/Scripts/activate      # Windows
pip install -r requirements.txt   # if you add a requirements file

##How to Use

Train the model
python src/train.py

Predict on new data
python src/predict.py --input data/raw/power_data.csv --output data/predictions.csv

Generate EDA visualizations
python src/eda_visualization.py

Plots will be saved in the plots/ folder.
