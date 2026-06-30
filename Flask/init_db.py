import datetime
import os
import sys

# Ensure models module can be imported if script is run directly
sys.path.append(os.path.dirname(__file__))

from models import init_db, SessionLocal, User, Country, Dataset, MLModel

def seed_database():
    # Initialize schema
    init_db()
    
    db = SessionLocal()
    try:
        # Check if database is already seeded
        if db.query(User).first() is not None:
            print("Database already seeded.")
            return

        print("Seeding database...")

        # 1. Seed Users
        users = [
            User(name="Alice Smith", email="alice.smith@wellbeing.org", role="Analyst"),
            User(name="Bob Johnson", email="bob.johnson@wellbeing.org", role="Data Scientist"),
            User(name="Charlie Brown", email="charlie.brown@wellbeing.org", role="Viewer")
        ]
        db.add_all(users)
        db.commit()

        # 2. Seed Countries
        countries = [
            Country(country_name="Switzerland", region="Europe", population=8700000),
            Country(country_name="Norway", region="Europe", population=5400000),
            Country(country_name="Iceland", region="Europe", population=370000),
            Country(country_name="Hong Kong", region="Asia", population=7400000),
            Country(country_name="Australia", region="Oceania", population=25600000),
            Country(country_name="United States", region="Americas", population=331000000),
            Country(country_name="Japan", region="Asia", population=125800000),
            Country(country_name="Germany", region="Europe", population=83200000),
            Country(country_name="Brazil", region="Americas", population=214300000),
            Country(country_name="India", region="Asia", population=1408000000),
            Country(country_name="Egypt", region="Africa", population=109300000),
            Country(country_name="Kenya", region="Africa", population=54000000),
            Country(country_name="Niger", region="Africa", population=25000000)
        ]
        db.add_all(countries)
        db.commit()

        # 3. Seed Dataset
        dataset = Dataset(
            dataset_name="UNDP Human Development Report 2024",
            source="United Nations Development Programme (UNDP)",
            total_rows=195,
            total_columns=14
        )
        db.add(dataset)
        db.commit()

        # 4. Seed ML Models
        models = [
            MLModel(
                dataset_id=dataset.dataset_id,
                model_name="RandomForestRegressor (HDI Score Predictor)",
                algorithm_used="Random Forest Regressor",
                accuracy_score=0.985,
                r2_score=0.978,
                model_file_path="Flask/HDI.pkl" # updated path
            ),
            MLModel(
                dataset_id=dataset.dataset_id,
                model_name="NeuralNetworkRegressor (Deep Well-Being)",
                algorithm_used="Multi-Layer Perceptron (MLP)",
                accuracy_score=0.962,
                r2_score=0.954,
                model_file_path="Flask/HDI.pkl" # fallback to same pickle file
            ),
            MLModel(
                dataset_id=dataset.dataset_id,
                model_name="LinearRegression (Baseline HDI)",
                algorithm_used="Linear Regression",
                accuracy_score=0.912,
                r2_score=0.898,
                model_file_path="Flask/HDI.pkl" # fallback
            )
        ]
        db.add_all(models)
        db.commit()

        print("Database seeding completed successfully.")

    except Exception as e:
        db.rollback()
        print(f"Error during seeding: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
