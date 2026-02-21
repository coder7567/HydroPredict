"""
HydroPredict: A Generalized High-Fidelity Swim Performance Engine
Complete production-ready implementation for elite swimming analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from abc import ABC, abstractmethod

# Model imports with fallbacks
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not available, falling back to sklearn models")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import joblib
import os
from datetime import datetime

# =============================================================================
# Data Layer - Core Domain Models
# =============================================================================

@dataclass
class BioMetrics:
    """Physiological metrics layer - modular athlete state representation"""
    hrv: float  # Heart Rate Variability (ms)
    resting_hr: float  # Resting Heart Rate (bpm)
    blood_lactate: Optional[float]  # mmol/L, optional
    sleep_hours: float  # Last 3 days average
    sleep_quality: float  # Score 0-100
    muscle_soreness: float  # 1-10 scale
    body_mass: float  # kg
    hydration_estimate: float  # % hydration level
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to normalized feature vector"""
        base = [
            self.hrv,
            self.resting_hr,
            self.sleep_hours,
            self.sleep_quality,
            self.muscle_soreness,
            self.body_mass,
            self.hydration_estimate
        ]
        # Handle optional lactate
        lactate_val = self.blood_lactate if self.blood_lactate is not None else 1.5
        return np.array(base + [lactate_val])


@dataclass
class Environment:
    """Environmental conditions with high precision"""
    water_temp: float  # Celsius
    air_temp: float  # Celsius
    humidity: float  # %
    barometric_pressure: float  # hPa
    altitude: float  # meters above sea level
    lane_number: int  # 1-8
    pool_type: str  # 'SCY', 'SCM', 'LCM'
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numerical features with encoding"""
        # Encode pool type
        pool_encodings = {'SCY': 0, 'SCM': 1, 'LCM': 2}
        pool_encoded = pool_encodings.get(self.pool_type, 1)
        
        return np.array([
            self.water_temp,
            self.air_temp,
            self.humidity,
            self.barometric_pressure,
            self.altitude,
            self.lane_number,
            pool_encoded
        ])


@dataclass
class StrokeModel:
    """Hydrodynamic modeling layer - stroke efficiency metrics"""
    stroke_rate: float  # strokes per minute
    stroke_length: float  # meters per stroke
    distance_per_stroke: float  # meters
    turn_time: float  # seconds
    underwater_distance: float  # meters
    intra_cycle_velocity_fluctuation: float  # ICVF - velocity variation within stroke cycle
    
    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.stroke_rate,
            self.stroke_length,
            self.distance_per_stroke,
            self.turn_time,
            self.underwater_distance,
            self.intra_cycle_velocity_fluctuation
        ])
    
    @property
    def efficiency_score(self) -> float:
        """Calculate composite efficiency score"""
        # Higher stroke length and lower ICVF = more efficient
        length_efficiency = min(self.stroke_length / 2.5, 1.0)  # Normalize to world-class
        icvf_efficiency = max(0, 1 - self.intra_cycle_velocity_fluctuation / 0.5)
        turn_efficiency = max(0, 1 - self.turn_time / 3.0)
        
        return np.mean([length_efficiency, icvf_efficiency, turn_efficiency]) * 100


@dataclass
class CognitiveMetrics:
    """Cognitive/behavioral layer - mental readiness indicators"""
    stress_level: float  # 1-10 scale
    screen_time: float  # hours last 24h
    gaming_hours: float  # hours last 24h
    focus_rating: float  # 1-10 scale
    pre_race_anxiety: float  # 1-10 scale
    
    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.stress_level,
            self.screen_time,
            self.gaming_hours,
            self.focus_rating,
            self.pre_race_anxiety
        ])
    
    @property
    def cognitive_load_index(self) -> float:
        """Calculate CLI - composite cognitive load metric"""
        return (self.stress_level * 0.3 + 
                (self.screen_time / 24) * 20 * 0.2 +
                (self.gaming_hours / 8) * 10 * 0.2 +
                (10 - self.focus_rating) * 0.15 +
                self.pre_race_anxiety * 0.15)


@dataclass
class SwimmerProfile:
    """Complete swimmer profile with all metrics"""
    name: str
    event: str
    last_time: float  # seconds
    biometrics: BioMetrics
    environment: Environment
    stroke: StrokeModel
    cognitive: CognitiveMetrics
    weekly_yardage: float  # meters
    training_load: float  # arbitrary units
    
    def to_feature_matrix(self) -> np.ndarray:
        """Concatenate all features into single vector"""
        return np.concatenate([
            [self.last_time, self.weekly_yardage, self.training_load],
            self.biometrics.to_feature_vector(),
            self.environment.to_feature_vector(),
            self.stroke.to_feature_vector(),
            self.cognitive.to_feature_vector()
        ])


# =============================================================================
# Model Layer - Machine Learning Architecture
# =============================================================================

class BaseModel(ABC):
    """Abstract base for all prediction models"""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        pass


class XGBoostModel(BaseModel):
    """XGBoost implementation with optimal parameters"""
    
    def __init__(self, **kwargs):
        if XGB_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **kwargs
            )
        else:
            raise ImportError("XGBoost not available")
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))


class GradientBoostingModel(BaseModel):
    """Sklearn Gradient Boosting fallback"""
    
    def __init__(self, **kwargs):
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            **kwargs
        )
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))


class RandomForestModel(BaseModel):
    """Random Forest fallback"""
    
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            **kwargs
        )
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))


class ModelFactory:
    """Factory for creating appropriate model instances"""
    
    @staticmethod
    def create_model(model_type: str = 'auto') -> BaseModel:
        if model_type == 'xgboost' and XGB_AVAILABLE:
            return XGBoostModel()
        elif model_type == 'gradient_boosting':
            return GradientBoostingModel()
        elif model_type == 'random_forest':
            return RandomForestModel()
        elif model_type == 'auto':
            if XGB_AVAILABLE:
                return XGBoostModel()
            else:
                try:
                    return GradientBoostingModel()
                except:
                    return RandomForestModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Data Pipeline Layer
# =============================================================================

class DataIngestionPipeline:
    """Handles CSV import and data validation"""
    
    REQUIRED_COLUMNS = [
        'name', 'event', 'last_time', 'hrv', 'resting_hr', 'sleep_hours',
        'sleep_quality', 'muscle_soreness', 'body_mass', 'hydration',
        'water_temp', 'air_temp', 'humidity', 'pressure', 'altitude',
        'lane', 'pool_type', 'stroke_rate', 'stroke_length',
        'distance_per_stroke', 'turn_time', 'underwater_distance', 'icvf',
        'stress', 'screen_time', 'gaming_hours', 'focus', 'anxiety',
        'weekly_yardage', 'training_load'
    ]
    
    @staticmethod
    def load_from_csv(filepath: str) -> List[SwimmerProfile]:
        """Load and parse CSV into SwimmerProfile objects"""
        df = pd.read_csv(filepath)
        
        # Validate columns
        missing = set(DataIngestionPipeline.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        profiles = []
        for _, row in df.iterrows():
            profile = DataIngestionPipeline._row_to_profile(row)
            profiles.append(profile)
        
        return profiles
    
    @staticmethod
    def _row_to_profile(row: pd.Series) -> SwimmerProfile:
        """Convert DataFrame row to SwimmerProfile"""
        
        biometrics = BioMetrics(
            hrv=row['hrv'],
            resting_hr=row['resting_hr'],
            blood_lactate=row.get('lactate'),  # optional
            sleep_hours=row['sleep_hours'],
            sleep_quality=row['sleep_quality'],
            muscle_soreness=row['muscle_soreness'],
            body_mass=row['body_mass'],
            hydration_estimate=row['hydration']
        )
        
        environment = Environment(
            water_temp=row['water_temp'],
            air_temp=row['air_temp'],
            humidity=row['humidity'],
            barometric_pressure=row['pressure'],
            altitude=row['altitude'],
            lane_number=row['lane'],
            pool_type=row['pool_type']
        )
        
        stroke = StrokeModel(
            stroke_rate=row['stroke_rate'],
            stroke_length=row['stroke_length'],
            distance_per_stroke=row['distance_per_stroke'],
            turn_time=row['turn_time'],
            underwater_distance=row['underwater_distance'],
            intra_cycle_velocity_fluctuation=row['icvf']
        )
        
        cognitive = CognitiveMetrics(
            stress_level=row['stress'],
            screen_time=row['screen_time'],
            gaming_hours=row['gaming_hours'],
            focus_rating=row['focus'],
            pre_race_anxiety=row['anxiety']
        )
        
        return SwimmerProfile(
            name=row['name'],
            event=row['event'],
            last_time=row['last_time'],
            biometrics=biometrics,
            environment=environment,
            stroke=stroke,
            cognitive=cognitive,
            weekly_yardage=row['weekly_yardage'],
            training_load=row['training_load']
        )


class FeatureEngineeringPipeline:
    """Feature engineering and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
    
    def fit_transform(self, profiles: List[SwimmerProfile]) -> np.ndarray:
        """Fit scaler and transform profiles to feature matrix"""
        X = self._profiles_to_matrix(profiles)
        self.feature_names = self._generate_feature_names()
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled
    
    def transform(self, profiles: Union[SwimmerProfile, List[SwimmerProfile]]) -> np.ndarray:
        """Transform profiles using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        if isinstance(profiles, SwimmerProfile):
            profiles = [profiles]
        
        X = self._profiles_to_matrix(profiles)
        return self.scaler.transform(X)
    
    def _profiles_to_matrix(self, profiles: List[SwimmerProfile]) -> np.ndarray:
        """Convert list of profiles to feature matrix"""
        matrices = [p.to_feature_matrix() for p in profiles]
        return np.array(matrices)
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for interpretability"""
        base_features = ['last_time', 'weekly_yardage', 'training_load']
        bio_features = ['hrv', 'resting_hr', 'sleep_hours', 'sleep_quality', 
                       'muscle_soreness', 'body_mass', 'hydration', 'lactate']
        env_features = ['water_temp', 'air_temp', 'humidity', 'pressure', 
                       'altitude', 'lane', 'pool_type_encoded']
        stroke_features = ['stroke_rate', 'stroke_length', 'distance_per_stroke',
                          'turn_time', 'underwater_distance', 'icvf']
        cog_features = ['stress', 'screen_time', 'gaming_hours', 'focus', 'anxiety']
        
        return base_features + bio_features + env_features + stroke_features + cog_features


# =============================================================================
# Prediction Engine
# =============================================================================

class SwimPredictor:
    """Core prediction engine with training and inference capabilities"""
    
    def __init__(self, model_type: str = 'auto'):
        self.model = ModelFactory.create_model(model_type)
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.training_history = {}
        
    def train(self, profiles: List[SwimmerProfile], target: str = 'last_time'):
        """Train the model on historical data"""
        # Prepare features
        X = self.feature_pipeline.fit_transform(profiles)
        y = np.array([getattr(p, target) for p in profiles])
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.training_history = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'n_samples': len(profiles)
        }
        
        return self.training_history
    
    def predict(self, profile: SwimmerProfile) -> float:
        """Predict performance for a single profile"""
        X = self.feature_pipeline.transform(profile)
        return float(self.model.predict(X)[0])
    
    def predict_batch(self, profiles: List[SwimmerProfile]) -> np.ndarray:
        """Predict for multiple profiles"""
        X = self.feature_pipeline.transform(profiles)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings"""
        if not self.feature_pipeline.feature_names:
            raise ValueError("Model must be trained first")
        
        return self.model.feature_importance(self.feature_pipeline.feature_names)


class MonteCarloSimulator:
    """Monte Carlo simulation for performance distribution analysis"""
    
    def __init__(self, n_runs: int = 10000, variability: float = 0.2):
        self.n_runs = n_runs
        self.variability = variability
        self.random_state = np.random.RandomState(42)
    
    def simulate(self, base_prediction: float, 
                 fatigue_factor: float = 1.0,
                 environmental_uncertainty: float = 0.1) -> Dict[str, float]:
        """
        Run Monte Carlo simulation with realistic variance components
        
        Args:
            base_prediction: Predicted race time
            fatigue_factor: Multiplier for fatigue effects
            environmental_uncertainty: Additional variance from environment
        
        Returns:
            Dictionary with simulation statistics
        """
        # Combine variance sources
        total_variance = self.variability * fatigue_factor + environmental_uncertainty
        
        # Generate simulations
        simulations = self.random_state.normal(
            base_prediction, 
            total_variance, 
            self.n_runs
        )
        
        # Ensure non-negative times
        simulations = np.maximum(simulations, base_prediction * 0.7)
        
        return {
            'expected': float(np.mean(simulations)),
            'hard_cap': float(np.percentile(simulations, 1)),  # Near theoretical limit
            'panic_threshold': float(np.percentile(simulations, 90)),  # Upper bound
            'consistency': float(np.std(simulations)),
            'min_possible': float(np.min(simulations)),
            'max_possible': float(np.max(simulations)),
            'simulations': simulations.tolist()  # For further analysis
        }


# =============================================================================
# Report Generation
# =============================================================================

class PerformanceReport:
    """Generates comprehensive performance certificates"""
    
    def __init__(self, predictor: SwimPredictor, simulator: MonteCarloSimulator):
        self.predictor = predictor
        self.simulator = simulator
    
    def generate_certificate(self, profile: SwimmerProfile) -> Dict:
        """Generate complete performance certificate"""
        
        # Get base prediction
        pred_time = self.predictor.predict(profile)
        
        # Run Monte Carlo simulation
        simulation_results = self.simulator.simulate(
            pred_time,
            fatigue_factor=profile.training_load / 100,  # Normalize fatigue
            environmental_uncertainty=0.05 if profile.environment.pool_type == 'LCM' else 0.08
        )
        
        # Calculate efficiency metrics
        efficiency_score = profile.stroke.efficiency_score
        cognitive_load = profile.cognitive.cognitive_load_index
        
        # Get feature importance
        feature_importance = self.predictor.get_feature_importance()
        
        # Calculate physiological red-zone threshold
        # Based on HR and lactate thresholds
        hr_threshold = 180 - profile.biometrics.resting_hr * 0.5
        red_zone_threshold = pred_time * (1 + (hr_threshold - 160) / 400)
        
        # Compile certificate
        certificate = {
            'athlete': profile.name,
            'event': profile.event,
            'timestamp': datetime.now().isoformat(),
            
            # Performance metrics
            'predicted_time': pred_time,
            'physiological_hard_cap': simulation_results['hard_cap'],
            'red_zone_threshold': red_zone_threshold,
            'efficiency_score': efficiency_score,
            'cognitive_load_index': cognitive_load,
            
            # Simulation statistics
            'simulation': {
                'expected': simulation_results['expected'],
                'consistency': simulation_results['consistency'],
                'confidence_interval': [
                    simulation_results['hard_cap'],
                    simulation_results['panic_threshold']
                ]
            },
            
            # Training context
            'training_load': profile.training_load,
            'weekly_yardage': profile.weekly_yardage,
            
            # Explainable AI
            'top_features': sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            
            # Model confidence
            'model_metrics': self.predictor.training_history if self.predictor.training_history else {}
        }
        
        return certificate
    
    def print_certificate(self, certificate: Dict):
        """Pretty print performance certificate"""
        print("\n" + "="*60)
        print(f"🏊 HYDROPREDICT PERFORMANCE CERTIFICATE")
        print("="*60)
        print(f"Athlete: {certificate['athlete']}")
        print(f"Event: {certificate['event']}")
        print(f"Generated: {certificate['timestamp']}")
        print("-"*60)
        
        print(f"\n🏁 Predicted Time: {certificate['predicted_time']:.2f}s")
        print(f"🧱 Physiological Hard Cap: {certificate['physiological_hard_cap']:.2f}s")
        print(f"🔴 Red-Zone Threshold: {certificate['red_zone_threshold']:.2f}s")
        print(f"⚙️  Efficiency Score: {certificate['efficiency_score']:.1f}/100")
        print(f"🧠 Cognitive Load: {certificate['cognitive_load_index']:.2f}")
        
        print(f"\n📊 Simulation Results ({self.simulator.n_runs:,} runs):")
        print(f"   Expected: {certificate['simulation']['expected']:.2f}s")
        print(f"   Consistency: ±{certificate['simulation']['consistency']:.3f}s")
        print(f"   90% CI: [{certificate['simulation']['confidence_interval'][0]:.2f}, "
              f"{certificate['simulation']['confidence_interval'][1]:.2f}]")
        
        print(f"\n📈 Top 5 Influencing Factors:")
        for i, (feature, importance) in enumerate(certificate['top_features'][:5], 1):
            print(f"   {i}. {feature}: {importance:.3f}")
        
        if certificate['model_metrics']:
            print(f"\n🤖 Model Performance:")
            print(f"   MAE: {certificate['model_metrics'].get('mae', 0):.3f}s")
            print(f"   R²: {certificate['model_metrics'].get('r2', 0):.3f}")
        
        print("="*60 + "\n")


# =============================================================================
# Main Application
# =============================================================================

class HydroPredictApp:
    """Main application orchestrating all components"""
    
    def __init__(self, model_type: str = 'auto'):
        self.predictor = SwimPredictor(model_type)
        self.simulator = MonteCarloSimulator(n_runs=10000, variability=0.2)
        self.report_generator = PerformanceReport(self.predictor, self.simulator)
        self.data_pipeline = DataIngestionPipeline()
        self.trained = False
    
    def load_and_train(self, csv_path: str):
        """Load data from CSV and train model"""
        print(f"Loading data from {csv_path}...")
        profiles = self.data_pipeline.load_from_csv(csv_path)
        
        print(f"Training on {len(profiles)} athlete profiles...")
        metrics = self.predictor.train(profiles)
        
        print(f"Training complete - MAE: {metrics['mae']:.3f}s, R²: {metrics['r2']:.3f}")
        self.trained = True
        
        return profiles
    
    def analyze_athlete(self, profile: SwimmerProfile) -> Dict:
        """Generate complete analysis for an athlete"""
        if not self.trained:
            raise ValueError("Model must be trained before analysis")
        
        return self.report_generator.generate_certificate(profile)
    
    def batch_analyze(self, profiles: List[SwimmerProfile]) -> List[Dict]:
        """Analyze multiple athletes"""
        return [self.analyze_athlete(p) for p in profiles]
    
    def export_report(self, certificate: Dict, filepath: str):
        """Export certificate to JSON"""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(certificate, f, default=convert_numpy, indent=2)


# =============================================================================
# Example Usage
# =============================================================================

def create_sample_data() -> str:
    """Create sample CSV data for demonstration"""
    import tempfile
    
    sample_data = pd.DataFrame({
        'name': ['Athlete_A', 'Athlete_B', 'Athlete_C', 'Athlete_D', 'Athlete_E'],
        'event': ['100m Free', '200m IM', '50m Free', '100m Back', '200m Free'],
        'last_time': [52.3, 128.5, 23.8, 58.2, 115.7],
        'hrv': [65, 58, 72, 61, 55],
        'resting_hr': [48, 52, 45, 50, 55],
        'sleep_hours': [8.2, 7.5, 8.5, 7.8, 7.2],
        'sleep_quality': [85, 78, 92, 82, 70],
        'muscle_soreness': [3, 5, 2, 4, 6],
        'body_mass': [75.2, 68.5, 80.1, 72.3, 70.5],
        'hydration': [98, 95, 97, 94, 92],
        'lactate': [1.2, 1.5, 1.1, 1.3, 1.8],
        'water_temp': [27.8, 27.5, 28.0, 27.6, 27.7],
        'air_temp': [24.5, 24.2, 24.8, 24.3, 24.4],
        'humidity': [55, 58, 52, 56, 57],
        'pressure': [1013, 1012, 1014, 1013, 1012],
        'altitude': [0, 0, 0, 0, 0],
        'lane': [4, 3, 5, 2, 4],
        'pool_type': ['LCM', 'LCM', 'SCM', 'LCM', 'SCY'],
        'stroke_rate': [52, 48, 65, 50, 55],
        'stroke_length': [2.1, 2.3, 1.9, 2.2, 2.4],
        'distance_per_stroke': [2.05, 2.25, 1.85, 2.15, 2.35],
        'turn_time': [0.8, 0.9, 0.7, 0.85, 0.95],
        'underwater_distance': [12.5, 10.2, 13.8, 11.5, 9.8],
        'icvf': [0.12, 0.15, 0.10, 0.13, 0.18],
        'stress': [3, 5, 2, 4, 6],
        'screen_time': [2.5, 4.0, 1.5, 3.0, 5.0],
        'gaming_hours': [0.5, 1.0, 0.0, 0.5, 2.0],
        'focus': [9, 7, 9, 8, 6],
        'anxiety': [3, 5, 2, 4, 7],
        'weekly_yardage': [35000, 32000, 38000, 34000, 30000],
        'training_load': [85, 78, 92, 82, 70]
    })
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    sample_data.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name


def main():
    """Example usage demonstrating complete pipeline"""
    print("🚀 HydroPredict: Elite Swim Performance Engine")
    print("Initializing system...")
    
    # Create app with best available model
    app = HydroPredictApp(model_type='auto')
    
    # Create sample data
    print("Creating sample dataset...")
    sample_csv = create_sample_data()
    
    # Load and train
    print("Loading and training model...")
    profiles = app.load_and_train("C:/Users/levir/OneDrive/Desktop/swimmers.csv")
    
    # Analyze a specific athlete
    print("\n🔬 Generating performance certificate for Athlete_C...")
    athlete_c = next(p for p in profiles if p.name == 'Athlete_C')
    certificate = app.analyze_athlete(athlete_c)
    
    # Print certificate
    app.report_generator.print_certificate(certificate)
    
    # Export to JSON
    app.export_report(certificate, 'athlete_c_certificate.json')
    print("Certificate exported to 'athlete_c_certificate.json'")
    
    # Batch analysis example
    print("\n📊 Batch analysis for all athletes:")
    all_certificates = app.batch_analyze(profiles)
    
    # Show summary statistics
    predicted_times = [c['predicted_time'] for c in all_certificates]
    print(f"Predicted times: {', '.join([f'{t:.2f}s' for t in predicted_times])}")
    
    # Cleanup
    os.unlink(sample_csv)
    
    print("\n✅ HydroPredict demonstration complete!")


if __name__ == "__main__":
    main()
