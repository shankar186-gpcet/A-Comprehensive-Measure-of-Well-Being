import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def main():
    # Load dataset
    csv_path = os.path.join(os.path.dirname(__file__), "..", "Dataset", "HDI.csv")
    if not os.path.exists(csv_path):
        print(f"Error: dataset file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Features and Target
    features = ["life_expectancy", "mean_years_schooling", "expected_years_schooling", "gnl_per_capita"]
    target = "hdi_score"

    X = df[features]
    y = df[target]

    # Model training
    print("Training ML model (RandomForestRegressor)...")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Save to Flask/HDI.pkl
    out_dir = os.path.join(os.path.dirname(__file__), "..", "Flask")
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, "HDI.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model successfully saved to {pkl_path}")

if __name__ == "__main__":
    main()
