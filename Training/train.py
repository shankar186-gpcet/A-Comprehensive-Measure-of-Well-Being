import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

    # Fill null values in X using the column mean
    X = X.fillna(X.mean())

    # Drop target null rows to ensure model fits correctly
    non_null_y_mask = Y.notna()
    X = X[non_null_y_mask]
    Y = Y[non_null_y_mask]

    # Split dataset into training and testing sets (80/20 split)
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train LinearRegression on training set
    print("Training ML model (LinearRegression) on training set...")
    temp_model = LinearRegression()
    temp_model.fit(X_train, Y_train)

    # Evaluate on test set
    Y_pred = temp_model.predict(X_test)
    print(f"R² Score on Test Set: {r2_score(Y_test, Y_pred):.6f}")
    print(f"MSE on Test Set: {mean_squared_error(Y_test, Y_pred):.6f}")

    # Train final model on all data for production deployment
    print("Training final ML model (LinearRegression) on all records...")
    model = LinearRegression()
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
