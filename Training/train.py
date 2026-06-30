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

    # Standardize column ordering to enable integer index selections
    columns_layout = [
        'ISO3',
        'Human Development Groups',
        'Country',
        'UNDP Developing Regions',
        'Human Development Index (2021)',
        'Life Expectancy at Birth (2021)',
        'Expected Years of Schooling (2021)',
        'Mean Years of Schooling (2021)',
        'Gross National Income Per Capita (2021)'
    ]
    df_layout = df[columns_layout]

    # X: Independent variables (indices 5, 6, 7, 8)
    X = df_layout.iloc[:, [5, 6, 7, 8]]
    # Y: Dependent variable (index 4)
    Y = df_layout.iloc[:, 4]

    # Count null values
    print("Null values in X before imputation:")
    print(X.isnull().sum())

    # Fill null values in X using the column mean
    X = X.fillna(X.mean())

    # Drop target null rows to ensure model fits correctly
    non_null_y_mask = Y.notna()
    X = X[non_null_y_mask]
    Y = Y[non_null_y_mask]

    # Model training
    print("Training ML model (RandomForestRegressor)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)

    # Save to Flask/HDI.pkl
    out_dir = os.path.join(os.path.dirname(__file__), "..", "Flask")
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, "HDI.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model successfully saved to {pkl_path}")

if __name__ == "__main__":
    main()
