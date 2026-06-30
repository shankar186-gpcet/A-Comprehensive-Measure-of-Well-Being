# Vanguard HDI - Human Development Index Forecasting System

Vanguard HDI is a premium predictive analytics platform designed to analyze, predict, and visualize national human development scores. The system integrates a scikit-learn machine learning pipeline with a relational database and a responsive, glassmorphism Flask web application.

---

## 1. Project Architecture & Structure

The repository is structured into isolated, maintainable directories:

```text
A-Comprehensive-Measure-of-Well-Being/
├── Dataset/
│   └── HDI.csv                 # Raw UNDP historical training dataset
├── Flask/
│   ├── templates/
│   │   ├── home.html           # Introduction page layout
│   │   ├── index.html          # Interactive workspace dashboard
│   │   └── predict.html        # Traditional form prediction result page
│   ├── static/
│   │   ├── css/ (index.css)    # Premium glassmorphism styling
│   │   ├── js/ (index.js)      # Frontend controller and AJAX engine
│   │   └── reports/            # Dynamically generated SVG reports
│   ├── app.py                  # Flask web server and prediction routes
│   ├── models.py               # SQLAlchemy ORM database models
│   ├── init_db.py              # Database creation and seeding script
│   ├── schema.sql              # Raw SQL DDL schema
│   ├── HDI.pkl                 # Pre-trained and serialized ML model
│   └── hdi_wellbeing.db        # Seeded SQLite database
├── Training/
│   ├── HumDevIndex.ipynb       # Jupyter Notebook for EDA & model training
│   └── train.py                # Command-line model training script
├── requirements.txt            # Python package dependencies
└── README.md                   # Documentation guide
```

---

## 2. Relational Database Schema & ER Diagram

The system operates on a relational SQLite database (`hdi_wellbeing.db`) mapped via SQLAlchemy ORM.

### Entities & Primary Keys
- **User** (`user_id`): Tracks professional accounts accessing the workspace.
- **Session** (`session_id`): Logs access history and user session state.
- **Country** (`country_id`): Stores reference country names, regions, and populations.
- **Dataset** (`dataset_id`): Tracks raw training files uploaded to the system.
- **ML Model** (`model_id`): Holds metadata for registered training engines.
- **HDI Input Data** (`input_id`): Saves development metrics entered by users.
- **HDI Prediction** (`prediction_id`): Maps input metrics to predicted scores and categories.
- **Visualization Report** (`report_id`): Records dynamic SVG charts generated for predictions.

### Relationships
- **User to Session**: One user can initiate multiple sessions (1 to Many).
- **User to HDI Input Data**: One user can submit multiple HDI prediction inputs (1 to Many).
- **Country to HDI Input Data**: One country can have multiple input records for analysis (1 to Many).
- **Dataset to ML Model**: One dataset can be used to train multiple models (1 to Many).
- **ML Model to HDI Prediction**: One model can generate multiple predictions (1 to Many).
- **HDI Input Data to HDI Prediction**: One input record generates exactly one prediction (1 to 1).
- **HDI Prediction to Visualization Report**: One prediction generates one SVG chart (1 to 1).

---

## 3. Machine Learning Pipeline

### Dataset Exploration & Visualization
Exploratory data analysis is executed in [HumDevIndex.ipynb](file:///c:/AI%20Projects/A-Comprehensive-Measure-of-Well-Being/Training/HumDevIndex.ipynb):
- **Uniqueness Check**: Validated that all 195 countries represent unique entries.
- **Strip Plot**: Analyzed *Mean Years of Schooling vs HDI Score* for the first 20 countries.
- **Scatter Plot**: Mapped *Life Expectancy at Birth vs HDI Score* along with a regression trendline.
- **Correlation Heatmap**: Visualized correlation coefficients between features to verify predictive strength.

### Feature & Target Selection
- **Independent Variables ($X$)**:
  - `Life Expectancy at Birth (2021)` (Index 5 of standardized layout)
  - `Expected Years of Schooling (2021)` (Index 6)
  - `Mean Years of Schooling (2021)` (Index 7)
  - `Gross National Income Per Capita (2021)` (Index 8)
- **Dependent Target Variable ($Y$)**:
  - `Human Development Index (2021)` (Index 4)

### Data Preprocessing & Training
- Null values in $X$ are imputed using respective **column means** (`X = X.fillna(X.mean())`).
- Missing values in $Y$ are dropped to prevent training distortion.
- Data is split into **80% training** and **20% testing** subsets.
- Model trained: **Linear Regression** model.
  - **Validation Metrics**: **R² Score: 0.9808** | **MSE: 0.0005**
- The final model is serialized and pickled to `Flask/HDI.pkl` for deployment.

---

## 4. Flask Web Application

The web backend serves routes handling interactive and traditional form inputs:

### Routes
- **`/` (GET)**: Renders `home.html`—a premium glassmorphism landing page introducing the HDI index components.
- **`/dashboard` (GET)**: Renders `index.html`—the main interactive analyzer containing sliders, country selectors, dynamic AJAX calculators, and historical log grids.
- **`/predict` (POST)**: Receives form inputs from `index.html` via traditional POST request, predicts the score, and renders `predict.html` displaying the rounded prediction.
- **`/api/auth/login` (POST)**: Initiates user sessions.
- **`/api/auth/logout/<session_id>` (POST)**: Concludes user sessions.
- **`/api/countries` (GET)**: Populates country template metadata.
- **`/api/models` (GET)**: Populates ML model metadata.
- **`/api/predict` (POST)**: AJAX prediction endpoint that computes HDI, logs results to the database, and creates interactive SVG reports in `/static/reports/`.
- **`/api/predictions` (GET)**: Retrieves past log history.

---

## 5. Installation & Usage Guide

### Prerequisites
- Python 3.10+
- Virtualenv

### Setup Instructions
1. **Initialize Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Initialize and Seed Database**:
   ```bash
   cd Flask
   python init_db.py
   cd ..
   ```
4. **Train and Pickle ML Model**:
   ```bash
   python Training/train.py
   ```
5. **Run Flask Server**:
   ```bash
   python Flask/app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:8000/`.
