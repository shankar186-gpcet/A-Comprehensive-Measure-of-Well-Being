-- SQL Schema for Human Development Index (HDI) Prediction Database (SQLite Compatible)

-- Enable Foreign Key support in SQLite (needs to be run per connection)
PRAGMA foreign_keys = ON;

-- 1. COUNTRY Table
CREATE TABLE IF NOT EXISTS country (
    country_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_name VARCHAR(255) NOT NULL,
    region VARCHAR(100) NOT NULL,
    population INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. USER Table
CREATE TABLE IF NOT EXISTS user (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 3. SESSION Table
CREATE TABLE IF NOT EXISTS session (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    login_time DATETIME NOT NULL,
    logout_time DATETIME,
    status VARCHAR(20) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user(user_id) ON DELETE CASCADE
);

-- 4. DATASET Table
CREATE TABLE IF NOT EXISTS dataset (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name VARCHAR(255) NOT NULL,
    source VARCHAR(255) NOT NULL,
    total_rows INTEGER,
    total_columns INTEGER,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 5. ML_MODEL Table
CREATE TABLE IF NOT EXISTS ml_model (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    algorithm_used VARCHAR(100) NOT NULL,
    accuracy_score REAL,
    r2_score REAL,
    model_file_path VARCHAR(255),
    trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id) ON DELETE SET NULL
);

-- 6. HDI_INPUT_DATA Table
CREATE TABLE IF NOT EXISTS hdi_input_data (
    input_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    country_id INTEGER NOT NULL,
    life_expectancy REAL NOT NULL,
    mean_years_schooling REAL NOT NULL,
    expected_years_schooling REAL NOT NULL,
    gnl_per_capita REAL NOT NULL, -- matching ER diagram 'gnl_per_capita'
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user(user_id) ON DELETE CASCADE,
    FOREIGN KEY (country_id) REFERENCES country(country_id) ON DELETE CASCADE
);

-- 7. HDI_PREDICTION Table
CREATE TABLE IF NOT EXISTS hdi_prediction (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_id INTEGER NOT NULL UNIQUE, -- 1:1 relationship with hdi_input_data
    model_id INTEGER NOT NULL,
    predicted_hdi_score REAL NOT NULL,
    hdi_category VARCHAR(50) NOT NULL,
    prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (input_id) REFERENCES hdi_input_data(input_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES ml_model(model_id) ON DELETE CASCADE
);

-- 8. VISUALIZATION_REPORT Table
CREATE TABLE IF NOT EXISTS visualization_report (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    graph_path VARCHAR(255) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES hdi_prediction(prediction_id) ON DELETE CASCADE
);
