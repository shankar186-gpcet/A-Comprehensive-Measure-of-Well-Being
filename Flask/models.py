import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, create_engine
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker

Base = declarative_base()

class Country(Base):
    __tablename__ = 'country'

    country_id = Column(Integer, primary_key=True, autoincrement=True)
    country_name = Column(String(255), nullable=False)
    region = Column(String(100), nullable=False)
    population = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    inputs = relationship("HDIInputData", back_populates="country", cascade="all, delete-orphan")

class User(Base):
    __tablename__ = 'user'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    role = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    inputs = relationship("HDIInputData", back_populates="user", cascade="all, delete-orphan")

class Session(Base):
    __tablename__ = 'session'

    session_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('user.user_id', ondelete='CASCADE'), nullable=False)
    login_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    logout_time = Column(DateTime)
    status = Column(String(20), nullable=False, default='Active')

    # Relationships
    user = relationship("User", back_populates="sessions")

class Dataset(Base):
    __tablename__ = 'dataset'

    dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(255), nullable=False)
    source = Column(String(255), nullable=False)
    total_rows = Column(Integer)
    total_columns = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    models = relationship("MLModel", back_populates="dataset")

class MLModel(Base):
    __tablename__ = 'ml_model'

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('dataset.dataset_id', ondelete='SET NULL'), nullable=False)
    model_name = Column(String(255), nullable=False)
    algorithm_used = Column(String(100), nullable=False)
    accuracy_score = Column(Float)
    r2_score = Column(Float)
    model_file_path = Column(String(255))
    trained_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="models")
    predictions = relationship("HDIPrediction", back_populates="model", cascade="all, delete-orphan")

class HDIInputData(Base):
    __tablename__ = 'hdi_input_data'

    input_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('user.user_id', ondelete='CASCADE'), nullable=False)
    country_id = Column(Integer, ForeignKey('country.country_id', ondelete='CASCADE'), nullable=False)
    life_expectancy = Column(Float, nullable=False)
    mean_years_schooling = Column(Float, nullable=False)
    expected_years_schooling = Column(Float, nullable=False)
    gnl_per_capita = Column(Float, nullable=False) # matching ER diagram 'gnl_per_capita'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="inputs")
    country = relationship("Country", back_populates="inputs")
    prediction = relationship("HDIPrediction", back_populates="input_data", uselist=False, cascade="all, delete-orphan")

class HDIPrediction(Base):
    __tablename__ = 'hdi_prediction'

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    input_id = Column(Integer, ForeignKey('hdi_input_data.input_id', ondelete='CASCADE'), unique=True, nullable=False)
    model_id = Column(Integer, ForeignKey('ml_model.model_id', ondelete='CASCADE'), nullable=False)
    predicted_hdi_score = Column(Float, nullable=False)
    hdi_category = Column(String(50), nullable=False)
    prediction_time = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    input_data = relationship("HDIInputData", back_populates="prediction")
    model = relationship("MLModel", back_populates="predictions")
    reports = relationship("VisualizationReport", back_populates="prediction", cascade="all, delete-orphan")

class VisualizationReport(Base):
    __tablename__ = 'visualization_report'

    report_id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('hdi_prediction.prediction_id', ondelete='CASCADE'), nullable=False)
    graph_path = Column(String(255), nullable=False)
    report_type = Column(String(50), nullable=False)
    generated_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    prediction = relationship("HDIPrediction", back_populates="reports")

# Database Connection setup helper
DATABASE_URL = "sqlite:///./hdi_wellbeing.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
