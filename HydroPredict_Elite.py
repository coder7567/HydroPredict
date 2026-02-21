"""
HydroPredict Elite: High-Fidelity Swim Performance Engine with Advanced Sports Science
Version 2.0 - Production-ready with proper modeling rigor
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import json
import joblib
import os
from pathlib import Path

# Model imports with fallbacks
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not available, falling back to sklearn models")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# =============================================================================
# Domain Enums and Constants
# =============================================================================

class EventType(Enum):
    """Event classification for specialized modeling"""
    SPRINT_50 = "50m"
    SPRINT_100 = "100m"
    MID_DISTANCE_200 = "200m"
    DISTANCE_400 = "400m"
    DISTANCE_800 = "800m"
    DISTANCE_1500 = "1500m"
    IM_200 = "200m IM"
    IM_400 = "400m IM"
    
    @classmethod
    def get_category(cls, event: str) -> str:
        """Categorize event into broader groups"""
        event_lower = event.lower()
        if '50' in event_lower or '100' in event_lower:
            return 'SPRINT'
        elif '200' in event_lower and 'im' not in event_lower:
            return 'MID_DISTANCE'
        elif any(x in event_lower for x in ['400', '800', '1500']) and 'im' not in event_lower:
            return 'DISTANCE'
        elif 'im' in event_lower:
            return 'IM'
        return 'OTHER'


class PoolType(Enum):
    """Pool configuration types"""
    SCY = "SCY"  # Short Course Yards
    SCM = "SCM"  # Short Course Meters
    LCM = "LCM"  # Long Course Meters


# =============================================================================
# Advanced Data Models with Proper Separation
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
    
    def to_feature_vector(self, normalizers: Dict = None) -> np.ndarray:
        """Convert to normalized feature vector with dynamic scaling"""
        base = [
            self.hrv,
            self.resting_hr,
            self.sleep_hours,
            self.sleep_quality,
            self.muscle_soreness,
            self.body_mass,
            self.hydration_estimate
        ]
        
        # Apply dynamic normalization if available
        if normalizers:
            base = self._apply_normalization(base, normalizers)
        
        # Handle optional lactate
        lactate_val = self.blood_lactate if self.blood_lactate is not None else 1.5
        return np.array(base + [lactate_val])
    
    def _apply_normalization(self, values: List, normalizers: Dict) -> List:
        """Apply percentile-based normalization"""
        normalized = []
        for i, (key, val) in enumerate(zip(
            ['hrv', 'resting_hr', 'sleep_hours', 'sleep_quality', 
             'muscle_soreness', 'body_mass', 'hydration'], values
        )):
            if key in normalizers:
                min_val, max_val = normalizers[key]
                if max_val > min_val:
                    normalized.append((val - min_val) / (max_val - min_val))
                else:
                    normalized.append(0.5)
            else:
                normalized.append(val)
        return normalized


@dataclass
class Environment:
    """Environmental conditions with high precision"""
    water_temp: float  # Celsius
    air_temp: float  # Celsius
    humidity: float  # %
    barometric_pressure: float  # hPa
    altitude: float  # meters above sea level
    lane_number: int  # 1-8
    pool_type: PoolType
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numerical features with encoding"""
        pool_encoded = list(PoolType).index(self.pool_type)
        
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
    
    def to_feature_vector(self, normalizers: Dict = None) -> np.ndarray:
        """Convert to feature vector with dynamic normalization"""
        values = [
            self.stroke_rate,
            self.stroke_length,
            self.distance_per_stroke,
            self.turn_time,
            self.underwater_distance,
            self.intra_cycle_velocity_fluctuation
        ]
        
        if normalizers:
            values = self._apply_normalization(values, normalizers)
        
        return np.array(values)
    
    def _apply_normalization(self, values: List, normalizers: Dict) -> List:
        """Apply percentile-based normalization"""
        normalized = []
        for i, (key, val) in enumerate(zip(
            ['stroke_rate', 'stroke_length', 'distance_per_stroke', 
             'turn_time', 'underwater_distance', 'icvf'], values
        )):
            if key in normalizers:
                min_val, max_val = normalizers[key]
                if max_val > min_val:
                    normalized.append((val - min_val) / (max_val - min_val))
                else:
                    normalized.append(0.5)
            else:
                normalized.append(val)
        return normalized
    
    def efficiency_score(self, reference_values: Dict = None) -> float:
        """Calculate composite efficiency score with dynamic reference"""
        if reference_values:
            max_stroke_length = reference_values.get('stroke_length', (0, 2.5))[1]
            min_icvf = reference_values.get('icvf', (0, 0.1))[0]
            min_turn_time = reference_values.get('turn_time', (0, 0.5))[0]
        else:
            max_stroke_length = 2.5
            min_icvf = 0.1
            min_turn_time = 0.5
        
        length_efficiency = min(self.stroke_length / max_stroke_length, 1.0)
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
class TrainingLoad:
    """Advanced training load modeling with ACWR"""
    acute_load: float  # 7-day rolling average
    chronic_load: float  # 28-day rolling average
    weekly_yardage: float  # meters
    training_stress_balance: float  # -10 to 10 scale
    
    @property
    def acwr(self) -> float:
        """Acute:Chronic Workload Ratio"""
        if self.chronic_load > 0:
            return self.acute_load / self.chronic_load
        return 1.0
    
    @property
    def fatigue_factor(self) -> float:
        """Calculate fatigue multiplier based on ACWR"""
        # ACWR sweet spot: 0.8-1.3, danger zone: >1.5
        if self.acwr < 0.8:
            return 0.95  # Slightly positive (underloading)
        elif self.acwr <= 1.3:
            return 1.0  # Optimal
        elif self.acwr <= 1.5:
            return 1.05  # Mild fatigue
        else:
            return 1.0 + (self.acwr - 1.5) * 0.2  # Exponential fatigue
    
    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.acute_load,
            self.chronic_load,
            self.acwr,
            self.weekly_yardage,
            self.training_stress_balance
        ])


@dataclass
class SwimmerProfile:
    """Complete swimmer profile with proper time separation"""
    name: str
    event: str
    previous_best_time: float  # Historical best (seconds)
    target_time: float  # Future target (seconds)
    biometrics: BioMetrics
    environment: Environment
    stroke: StrokeModel
    cognitive: CognitiveMetrics
    training: TrainingLoad
    event_category: str = field(init=False)
    
    def __post_init__(self):
        self.event_category = EventType.get_category(self.event)
    
    def to_feature_matrix(self, normalizers: Dict = None) -> np.ndarray:
        """Concatenate all features into single vector"""
        return np.concatenate([
            [self.previous_best_time],
            self.biometrics.to_feature_vector(normalizers.get('bio', {}) if normalizers else None),
            self.environment.to_feature_vector(),
            self.stroke.to_feature_vector(normalizers.get('stroke', {}) if normalizers else None),
            self.cognitive.to_feature_vector(),
            self.training.to_feature_vector(),
            [hash(self.event_category) % 100]  # Simple event category encoding
        ])


# =============================================================================
# Ensemble Model Architecture
# =============================================================================

class EnsembleModel(ABC):
    """Ensemble of models for uncertainty estimation"""
    
    def __init__(self, n_estimators: int = 5, model_type: str = 'auto'):
        self.n_estimators = n_estimators
        self.models = []
        self.model_type = model_type
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble with bootstrap resampling for diversity"""
        self.models = []
        rng = np.random.RandomState(42)

        n = X.shape[0]

        for i in range(self.n_estimators):
            # Bootstrap sample (sample with replacement)
            idx = rng.randint(0, n, size=n)

            model = ModelFactory.create_model(self.model_type, random_state=42 + i)
            model.fit(X[idx], y[idx])
            self.models.append(model)

        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            mean_predictions: Average of all models
            uncertainty: Standard deviation across models (epistemic uncertainty)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        all_predictions = np.array([model.predict(X) for model in self.models])
        mean_pred = np.mean(all_predictions, axis=0)
        uncertainty = np.std(all_predictions, axis=0)
        
        return mean_pred, uncertainty
    
    def feature_importance(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Calculate permutation importance using validation set"""
        if not self.feature_names:
            raise ValueError("Feature names must be set")
        
        # Use first model for importance (ensemble would be too expensive)
        result = permutation_importance(
            self.models[0], X_val, y_val,
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        return dict(zip(self.feature_names, result.importances_mean))


class ModelFactory:
    """Factory for creating appropriate model instances"""
    
    @staticmethod
    def create_model(model_type: str = 'auto', random_state: int = 42) -> Any:
        if model_type == 'xgboost' and XGB_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state
            )
        elif model_type == 'gradient_boosting' or (model_type == 'auto' and not XGB_AVAILABLE):
            return GradientBoostingRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_state
            )
        elif model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            )
        elif model_type == 'auto' and XGB_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Advanced Data Pipeline
# =============================================================================

class DynamicDataIngestionPipeline:
    """Flexible data ingestion with dynamic schema mapping"""
    
    # Dynamic column prefixes for automatic mapping
    PREFIX_MAPPING = {
        'bio_': 'biometrics',
        'env_': 'environment',
        'stroke_': 'stroke',
        'cog_': 'cognitive',
        'train_': 'training'
    }
    
    def __init__(self, schema_file: Optional[str] = None):
        self.schema = self._load_schema(schema_file) if schema_file else {}
        self.label_encoders = {}
        
    def _load_schema(self, schema_file: str) -> Dict:
        """Load custom schema mapping from JSON"""
        with open(schema_file, 'r') as f:
            return json.load(f)
    
    def load_from_csv(self, filepath: str) -> List[SwimmerProfile]:
        """Load and parse CSV with dynamic column mapping"""
        df = pd.read_csv(filepath)
        
        # Detect column types based on prefixes
        column_mapping = self._map_columns(df.columns)
        
        profiles = []
        for _, row in df.iterrows():
            profile = self._row_to_profile(row, column_mapping)
            profiles.append(profile)
        
        return profiles
    
    def _map_columns(self, columns: List[str]) -> Dict:
        """Map column names to profile attributes"""
        mapping = {
            'biometrics': [],
            'environment': [],
            'stroke': [],
            'cognitive': [],
            'training': [],
            'core': []
        }
        
        for col in columns:
            mapped = False
            for prefix, category in self.PREFIX_MAPPING.items():
                if col.startswith(prefix):
                    mapping[category].append(col)
                    mapped = True
                    break
            
            if not mapped and col in ['name', 'event', 'previous_best_time', 'target_time']:
                mapping['core'].append(col)
        
        return mapping
    
    def _row_to_profile(self, row: pd.Series, mapping: Dict) -> SwimmerProfile:
        """Convert DataFrame row to SwimmerProfile using mapping"""
        
        # Core fields
        name = row.get('name', 'Unknown')
        event = row.get('event', 'Unknown')
        previous_best = float(row.get('previous_best_time', row.get('last_time', 0)))
        target = float(row.get('target_time', previous_best * 0.98))  # Default 2% improvement
        
        # Biometrics
        bio_kwargs = self._extract_prefix_fields(row, 'bio_', {
            'hrv': 65, 'resting_hr': 50, 'sleep_hours': 8, 'sleep_quality': 80,
            'muscle_soreness': 3, 'body_mass': 70, 'hydration_estimate': 95, 'blood_lactate': None
        })
        biometrics = BioMetrics(**bio_kwargs)
        
        # Environment
        env_kwargs = self._extract_prefix_fields(row, 'env_', {
            'water_temp': 27.5, 'air_temp': 24.0, 'humidity': 55,
            'barometric_pressure': 1013, 'altitude': 0, 'lane_number': 4,
            'pool_type': PoolType.LCM
        })
        # Convert pool_type string to Enum
        if isinstance(env_kwargs.get('pool_type'), str):
            env_kwargs['pool_type'] = PoolType(env_kwargs['pool_type'])
        environment = Environment(**env_kwargs)
        
        # Stroke
        stroke_kwargs = self._extract_prefix_fields(row, 'stroke_', {
            'stroke_rate': 55, 'stroke_length': 2.1, 'distance_per_stroke': 2.0,
            'turn_time': 0.8, 'underwater_distance': 12, 'intra_cycle_velocity_fluctuation': 0.15
        })
        stroke = StrokeModel(**stroke_kwargs)
        
        # Cognitive
        cog_kwargs = self._extract_prefix_fields(row, 'cog_', {
            'stress_level': 4, 'screen_time': 3, 'gaming_hours': 1,
            'focus_rating': 7, 'pre_race_anxiety': 4
        })
        cognitive = CognitiveMetrics(**cog_kwargs)
        
        # Training
        train_kwargs = self._extract_prefix_fields(row, 'train_', {
            'acute_load': 800, 'chronic_load': 750, 'weekly_yardage': 35000,
            'training_stress_balance': 0
        })
        training = TrainingLoad(**train_kwargs)
        
        return SwimmerProfile(
            name=name,
            event=event,
            previous_best_time=previous_best,
            target_time=target,
            biometrics=biometrics,
            environment=environment,
            stroke=stroke,
            cognitive=cognitive,
            training=training
        )
    
    def _extract_prefix_fields(self, row: pd.Series, prefix: str, defaults: Dict) -> Dict:
        """Extract fields with given prefix, falling back to defaults"""
        result = {}
        for field, default in defaults.items():
            col_name = f"{prefix}{field}"
            if col_name in row.index and pd.notna(row[col_name]):
                result[field] = row[col_name]
            else:
                result[field] = default
        return result


class AdvancedFeaturePipeline:
    """Feature engineering with dynamic normalization and event grouping"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.normalizers = {}  # For percentile-based normalization
        self.feature_names = []
        self.event_encoders = {}
        self.is_fitted = False
        
    def fit(self, profiles: List[SwimmerProfile]):
        """Calculate normalization parameters from training data"""
        
        # Calculate percentiles for dynamic normalization
        self._calculate_normalizers(profiles)
        
        # Transform and fit scaler
        X = self._profiles_to_matrix(profiles)
        self.feature_names = self._generate_feature_names()
        self.scaler.fit(X)
        self.is_fitted = True
        
        return self
    
    def transform(self, profiles: Union[SwimmerProfile, List[SwimmerProfile]]) -> np.ndarray:
        """Transform profiles using fitted parameters"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        if isinstance(profiles, SwimmerProfile):
            profiles = [profiles]
        
        X = self._profiles_to_matrix(profiles)
        return self.scaler.transform(X)
    
    def _calculate_normalizers(self, profiles: List[SwimmerProfile]):
        """Calculate percentile-based normalizers for key metrics"""
        
        # Collect values for each metric
        bio_values = {field: [] for field in 
                     ['hrv', 'resting_hr', 'sleep_hours', 'sleep_quality', 
                      'muscle_soreness', 'body_mass', 'hydration']}
        stroke_values = {field: [] for field in 
                        ['stroke_rate', 'stroke_length', 'distance_per_stroke',
                         'turn_time', 'underwater_distance', 'icvf']}
        
        for p in profiles:
            # Bio metrics
            bio_values['hrv'].append(p.biometrics.hrv)
            bio_values['resting_hr'].append(p.biometrics.resting_hr)
            bio_values['sleep_hours'].append(p.biometrics.sleep_hours)
            bio_values['sleep_quality'].append(p.biometrics.sleep_quality)
            bio_values['muscle_soreness'].append(p.biometrics.muscle_soreness)
            bio_values['body_mass'].append(p.biometrics.body_mass)
            bio_values['hydration'].append(p.biometrics.hydration_estimate)
            
            # Stroke metrics
            stroke_values['stroke_rate'].append(p.stroke.stroke_rate)
            stroke_values['stroke_length'].append(p.stroke.stroke_length)
            stroke_values['distance_per_stroke'].append(p.stroke.distance_per_stroke)
            stroke_values['turn_time'].append(p.stroke.turn_time)
            stroke_values['underwater_distance'].append(p.stroke.underwater_distance)
            stroke_values['icvf'].append(p.stroke.intra_cycle_velocity_fluctuation)
        
        # Calculate 5th and 95th percentiles for normalization
        self.normalizers = {
            'bio': {},
            'stroke': {}
        }
        
        for field, values in bio_values.items():
            if values:
                self.normalizers['bio'][field] = (
                    np.percentile(values, 5),
                    np.percentile(values, 95)
                )
        
        for field, values in stroke_values.items():
            if values:
                self.normalizers['stroke'][field] = (
                    np.percentile(values, 5),
                    np.percentile(values, 95)
                )
    
    def _profiles_to_matrix(self, profiles: List[SwimmerProfile]) -> np.ndarray:
        """Convert list of profiles to feature matrix with normalization"""
        matrices = []
        for p in profiles:
            # Use normalizers if available
            bio_norm = self.normalizers.get('bio') if self.is_fitted else None
            stroke_norm = self.normalizers.get('stroke') if self.is_fitted else None
            
            features = np.concatenate([
                [p.previous_best_time],
                p.biometrics.to_feature_vector(bio_norm),
                p.environment.to_feature_vector(),
                p.stroke.to_feature_vector(stroke_norm),
                p.cognitive.to_feature_vector(),
                p.training.to_feature_vector(),
                [hash(p.event_category) % 100]
            ])
            matrices.append(features)
        
        return np.array(matrices)
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for interpretability"""
        return [
            'previous_best_time',
            'hrv', 'resting_hr', 'sleep_hours', 'sleep_quality', 
            'muscle_soreness', 'body_mass', 'hydration', 'lactate',
            'water_temp', 'air_temp', 'humidity', 'pressure', 
            'altitude', 'lane', 'pool_type',
            'stroke_rate', 'stroke_length', 'distance_per_stroke',
            'turn_time', 'underwater_distance', 'icvf',
            'stress', 'screen_time', 'gaming_hours', 'focus', 'anxiety',
            'acute_load', 'chronic_load', 'acwr', 'weekly_yardage', 'tsb',
            'event_category'
        ]


# =============================================================================
# Enhanced Prediction Engine
# =============================================================================

class EnhancedSwimPredictor:
    """Advanced prediction engine with ensemble and uncertainty estimation"""
    
    def __init__(self, model_type: str = 'auto', n_ensemble: int = 5):
        self.ensemble = EnsembleModel(n_estimators=n_ensemble, model_type=model_type)
        self.feature_pipeline = AdvancedFeaturePipeline()
        self.training_history = {}
        self.model_version = "2.0.0"
        self.training_timestamp = None
        
        # Event-specific models
        self.event_models = {}
        
    def train(self, profiles: List[SwimmerProfile], use_event_specific: bool = True, target: str = 'target_time'):

        self.event_models = {}

        if use_event_specific:
            event_groups = {}
            for p in profiles:
                event_groups.setdefault(p.event_category, []).append(p)

            for category, cat_profiles in event_groups.items():
                if len(cat_profiles) >= 10:
                    self.event_models[category] = self._train_single(cat_profiles, target)
                else:
                    warnings.warn(f"Category {category} has only {len(cat_profiles)} samples, using general model")

        # Train + STORE the general model bundle
        general = self._train_single(profiles, target)
        self.ensemble = general["ensemble"]
        self.feature_pipeline = general["pipeline"]
        self.training_history = general["training_history"]
        self.feature_importance = general["feature_importance"]

        self.training_timestamp = datetime.now()
        return self.training_history
    
    def _train_single(self, profiles: List[SwimmerProfile], target: str):
        pipeline = AdvancedFeaturePipeline().fit(profiles)
        X = pipeline.transform(profiles)
        y = np.array([getattr(p, target) for p in profiles])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        ensemble = EnsembleModel(n_estimators=self.ensemble.n_estimators, model_type=self.ensemble.model_type)
        ensemble.feature_names = pipeline.feature_names
        ensemble.fit(X_train, y_train)

        y_pred, uncertainty = ensemble.predict(X_val)

        training_history = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred),
            'mean_uncertainty': float(np.mean(uncertainty)),
            'n_samples': len(profiles)
        }

        feature_importance = ensemble.feature_importance(X_val, y_val)

        return {
            "ensemble": ensemble,
            "pipeline": pipeline,
            "training_history": training_history,
            "feature_importance": feature_importance
        }
    
    def predict(self, profile: SwimmerProfile) -> Tuple[float, float]:
        """
        Predict with uncertainty
        Returns:
            prediction: Expected race time
            uncertainty: Model uncertainty (std dev)
        """
        X = self.feature_pipeline.transform(profile)
        
        # Try event-specific model first
        if profile.event_category in self.event_models:
            bundle = self.event_models[profile.event_category]
            X = bundle["pipeline"].transform(profile)
            pred, unc = bundle["ensemble"].predict(X)
        else:
            X = self.feature_pipeline.transform(profile)
            pred, unc = self.ensemble.predict(X)
        
        return float(pred[0]), float(unc[0])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings"""
        if not hasattr(self, 'feature_importance'):
            raise ValueError("Model must be trained first")
        
        return self.feature_importance
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'ensemble': self.ensemble,
            'feature_pipeline': self.feature_pipeline,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version,
            'training_timestamp': self.training_timestamp,
            'event_models': self.event_models
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EnhancedSwimPredictor':
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        predictor = cls()
        predictor.ensemble = model_data['ensemble']
        predictor.feature_pipeline = model_data['feature_pipeline']
        predictor.training_history = model_data['training_history']
        predictor.feature_importance = model_data['feature_importance']
        predictor.model_version = model_data['model_version']
        predictor.training_timestamp = model_data['training_timestamp']
        predictor.event_models = model_data.get('event_models', {})
        
        return predictor


class AdvancedMonteCarloSimulator:
    """Enhanced Monte Carlo with physiological and model uncertainty"""
    
    def __init__(self, n_runs: int = 10000):
        self.n_runs = n_runs
        self.random_state = np.random.RandomState(42)

    def physiological_limit_seconds(self, pred_time: float, profile: SwimmerProfile) -> float:
        """
        Absolute physiological limit (hard floor).
        Returns the fastest humanly possible time for THIS profile.
        """

        # HRV bonus (higher HRV -> more upside), clamp 0..1
        hrv_bonus = np.clip((profile.biometrics.hrv - 50.0) / 50.0, 0.0, 1.0)

        # Technique bonus from efficiency score (0..1)
        eff = profile.stroke.efficiency_score()
        eff_bonus = np.clip(eff / 100.0, 0.0, 1.0)

        # Fatigue penalty (fatigue_factor > 1 means tired); clamp 0..1
        fatigue_penalty = np.clip((profile.training.fatigue_factor - 1.0) * 2.0, 0.0, 1.0)

        # Improvement fraction: 3%..10% before fatigue penalty
        improvement = 0.03 + 0.07 * (0.6 * hrv_bonus + 0.4 * eff_bonus)

        # Fatigue reduces the achievable “best possible”
        improvement *= (1.0 - 0.7 * fatigue_penalty)

        # Safety clamp: never allow >12% improvement from prediction
        improvement = float(np.clip(improvement, 0.01, 0.12))

        return float(pred_time * (1.0 - improvement))
    
    def simulate(self, base_prediction: float,
                model_uncertainty: float,
                profile: SwimmerProfile) -> Dict[str, float]:
        # Physiological variability (aleatoric)
        physio_variance = 0.1 * (1 + (10 - profile.biometrics.hrv / 10) / 10)

        # Fatigue factor from ACWR
        fatigue_factor = profile.training.fatigue_factor

        # Environmental uncertainty
        env_uncertainty = 0.05 if profile.environment.pool_type == PoolType.LCM else 0.08

        # Combine all uncertainties
        total_std = np.sqrt(
            model_uncertainty**2 +
            physio_variance**2 +
            env_uncertainty**2
        ) * fatigue_factor

        # Absolute physiological limit (hard floor)
        absolute_limit = self.physiological_limit_seconds(base_prediction, profile)

        # "Perfect storm" shift
        perfect_day_shift = 0.25 * total_std

        # Draw RAW simulations first (unclamped)
        raw = self.random_state.normal(
            base_prediction - perfect_day_shift,
            total_std,
            self.n_runs
        )

        # Perfect storm from RAW distribution (rare best-case)
        perfect_storm = float(np.percentile(raw, 5.0))   # or 2.5

        # Clamp to hard floor for final outcomes
        ci_raw_low  = float(np.percentile(raw, 2.5))
        ci_raw_high = float(np.percentile(raw, 97.5))
        simulations = np.maximum(raw, absolute_limit)

        return {
            "expected": float(np.mean(simulations)),

            "absolute_physiological_limit": float(absolute_limit),

            # perfect storm can't beat the hard floor
            "perfect_storm_time": float(max(perfect_storm, absolute_limit)),

            "panic_threshold": float(np.percentile(simulations, 90)),
            "consistency": float(np.std(simulations)),
            "coefficient_of_variation": float(np.std(simulations) / np.mean(simulations)),
            "confidence_95_lower": float(np.percentile(simulations, 2.5)),
            "confidence_95_upper": float(np.percentile(simulations, 97.5)),
            "simulations": simulations.tolist()
        }

# =============================================================================
# Comprehensive Report Generator
# =============================================================================

class ComprehensiveReportGenerator:
    """Generates detailed performance certificates with all metrics"""
    
    def __init__(self, predictor: EnhancedSwimPredictor, simulator: AdvancedMonteCarloSimulator):
        self.predictor = predictor
        self.simulator = simulator
    
    def generate_certificate(self, profile: SwimmerProfile) -> Dict:
        """Generate complete performance certificate with all analytics"""
        
        # Get prediction with uncertainty
        pred_time, model_uncertainty = self.predictor.predict(profile)
        
        # Run Monte Carlo simulation
        simulation_results = self.simulator.simulate(
            pred_time,
            model_uncertainty,
            profile
        )
        
        # Calculate efficiency metrics with reference values
        stroke_ref = self.predictor.feature_pipeline.normalizers.get('stroke', {})
        efficiency_score = profile.stroke.efficiency_score(stroke_ref)
        
        # Cognitive metrics
        cognitive_load = profile.cognitive.cognitive_load_index
        
        # Red-zone = "blow-up risk time" (ALWAYS slower than predicted)
        lactate = profile.biometrics.blood_lactate if profile.biometrics.blood_lactate is not None else 4.0
        anxiety = profile.cognitive.pre_race_anxiety  # 1..10
        fatigue = profile.training.fatigue_factor     # ~0.95..1.2+

        slowdown = (
            0.02 +                           # base 2% slower
            0.01  * max(0.0, lactate - 4.0) +# lactate above 4 adds slowdown
            0.003 * max(0.0, anxiety - 5.0) +# anxiety above 5 adds slowdown
            0.04  * max(0.0, fatigue - 1.0)  # fatigue above 1 adds slowdown
        )

        slowdown = float(np.clip(slowdown, 0.02, 0.20))  # 2%..20% slower cap
        red_zone_threshold = pred_time * (1.0 + slowdown)
        
        # Feature importance
        feature_importance = self.predictor.get_feature_importance()
        
        # Training readiness
        acwr = profile.training.acwr
        if acwr < 0.8:
            readiness = "UNDERTRAINED"
        elif acwr <= 1.3:
            readiness = "OPTIMAL"
        elif acwr <= 1.5:
            readiness = "HIGH LOAD"
        else:
            readiness = "OVERTRAINING RISK"
        
        # Compile comprehensive certificate
        certificate = {
            'metadata': {
                'athlete': profile.name,
                'event': profile.event,
                'event_category': profile.event_category,
                'timestamp': datetime.now().isoformat(),
                'model_version': self.predictor.model_version,
                'training_date': self.predictor.training_timestamp.isoformat() if self.predictor.training_timestamp else None
            },
            
            'performance_metrics': {
                'predicted_time': pred_time,
                'previous_best': profile.previous_best_time,
                'target_time': profile.target_time,
                'improvement_potential': ((profile.previous_best_time - pred_time) / profile.previous_best_time) * 100,
                'absolute_physiological_limit': simulation_results['absolute_physiological_limit'],
                'perfect_storm_time': simulation_results['perfect_storm_time'],
                'red_zone_threshold': red_zone_threshold,
                'efficiency_score': efficiency_score,
                'cognitive_load_index': cognitive_load
            },
            
            'simulation': {
                'n_runs': self.simulator.n_runs,
                'expected': simulation_results['expected'],
                'consistency': simulation_results['consistency'],
                'cv': simulation_results['coefficient_of_variation'],
                'confidence_interval_95': [
                    simulation_results['confidence_95_lower'],
                    simulation_results['confidence_95_upper']
                ]
            },
            
            'training_analysis': {
                'acute_load': profile.training.acute_load,
                'chronic_load': profile.training.chronic_load,
                'acwr': acwr,
                'readiness_status': readiness,
                'fatigue_factor': profile.training.fatigue_factor,
                'weekly_yardage': profile.training.weekly_yardage
            },
            
            'physiological_state': {
                'hrv': profile.biometrics.hrv,
                'resting_hr': profile.biometrics.resting_hr,
                'sleep_quality': profile.biometrics.sleep_quality,
                'muscle_soreness': profile.biometrics.muscle_soreness,
                'hydration': profile.biometrics.hydration_estimate,
                'lactate': profile.biometrics.blood_lactate
            },
            
            'stroke_analysis': {
                'stroke_rate': profile.stroke.stroke_rate,
                'stroke_length': profile.stroke.stroke_length,
                'distance_per_stroke': profile.stroke.distance_per_stroke,
                'turn_time': profile.stroke.turn_time,
                'underwater_distance': profile.stroke.underwater_distance,
                'icvf': profile.stroke.intra_cycle_velocity_fluctuation
            },
            
            'cognitive_state': {
                'stress': profile.cognitive.stress_level,
                'screen_time': profile.cognitive.screen_time,
                'focus': profile.cognitive.focus_rating,
                'anxiety': profile.cognitive.pre_race_anxiety
            },
            
            'environmental_conditions': {
                'water_temp': profile.environment.water_temp,
                'air_temp': profile.environment.air_temp,
                'humidity': profile.environment.humidity,
                'altitude': profile.environment.altitude,
                'pool_type': profile.environment.pool_type.value
            },
            
            'model_confidence': {
                'prediction_uncertainty': model_uncertainty,
                'model_mae': self.predictor.training_history.get('mae', 0),
                'model_r2': self.predictor.training_history.get('r2', 0),
                'feature_importance': dict(sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:15])
            }
        }
        
        return certificate
    
    def print_certificate(self, certificate: Dict):
        """Pretty print comprehensive certificate"""
        print("\n" + "="*70)
        print(f"🏊 HYDROPREDICT ELITE PERFORMANCE CERTIFICATE v{self.predictor.model_version}")
        print("="*70)
        print(f"Athlete: {certificate['metadata']['athlete']}")
        print(f"Event: {certificate['metadata']['event']} ({certificate['metadata']['event_category']})")
        print(f"Generated: {certificate['metadata']['timestamp']}")
        print("-"*70)
        
        print(f"\n🏁 PERFORMANCE PREDICTION:")
        print(f"   Predicted Time: {certificate['performance_metrics']['predicted_time']:.2f}s")
        print(f"   Previous Best: {certificate['performance_metrics']['previous_best']:.2f}s")
        print(f"   Improvement: {certificate['performance_metrics']['improvement_potential']:.1f}%")
        print(f"   🧱 Absolute Physiological Limit: {certificate['performance_metrics']['absolute_physiological_limit']:.2f}s")
        print(f"   ⚡ Perfect Storm Scenario:      {certificate['performance_metrics']['perfect_storm_time']:.2f}s")
        print(f"   🔴 Red-Zone Threshold: {certificate['performance_metrics']['red_zone_threshold']:.2f}s")
        
        print(f"\n⚙️ EFFICIENCY ANALYSIS:")
        print(f"   Stroke Efficiency: {certificate['performance_metrics']['efficiency_score']:.1f}/100")
        print(f"   ICVF: {certificate['stroke_analysis']['icvf']:.3f}")
        print(f"   Turn Time: {certificate['stroke_analysis']['turn_time']:.2f}s")
        print(f"   Underwater: {certificate['stroke_analysis']['underwater_distance']:.1f}m")
        
        print(f"\n🧠 COGNITIVE LOAD:")
        print(f"   CLI: {certificate['performance_metrics']['cognitive_load_index']:.2f}")
        print(f"   Stress: {certificate['cognitive_state']['stress']}/10")
        print(f"   Focus: {certificate['cognitive_state']['focus']}/10")
        
        print(f"\n📊 TRAINING READINESS:")
        print(f"   ACWR: {certificate['training_analysis']['acwr']:.2f}")
        print(f"   Status: {certificate['training_analysis']['readiness_status']}")
        print(f"   Fatigue Factor: {certificate['training_analysis']['fatigue_factor']:.3f}")
        
        print(f"\n🎲 MONTE CARLO SIMULATION ({certificate['simulation']['n_runs']:,} runs):")
        print(f"   Expected: {certificate['simulation']['expected']:.2f}s")
        print(f"   Consistency (σ): ±{certificate['simulation']['consistency']:.3f}s")
        print(f"   CV: {certificate['simulation']['cv']:.3f}")
        print(f"   95% CI: [{certificate['simulation']['confidence_interval_95'][0]:.2f}, "
              f"{certificate['simulation']['confidence_interval_95'][1]:.2f}]")
        
        print(f"\n📈 TOP 10 INFLUENCING FACTORS:")
        for i, (feature, importance) in enumerate(
            list(certificate['model_confidence']['feature_importance'].items())[:10], 1
        ):
            print(f"   {i:2d}. {feature:25s}: {importance:+.4f}")
        
        print(f"\n🤖 MODEL METRICS:")
        print(f"   Prediction Uncertainty: ±{certificate['model_confidence']['prediction_uncertainty']:.3f}s")
        print(f"   Model MAE: {certificate['model_confidence']['model_mae']:.3f}s")
        print(f"   Model R²: {certificate['model_confidence']['model_r2']:.3f}")
        
        print("="*70 + "\n")


# =============================================================================
# Main Application with Enhanced Features
# =============================================================================

class HydroPredictEliteApp:
    """Main application with all advanced features"""
    
    def __init__(self, model_type: str = 'auto', n_ensemble: int = 5):
        self.predictor = EnhancedSwimPredictor(model_type=model_type, n_ensemble=n_ensemble)
        self.simulator = AdvancedMonteCarloSimulator(n_runs=10000)
        self.report_generator = ComprehensiveReportGenerator(self.predictor, self.simulator)
        self.data_pipeline = DynamicDataIngestionPipeline()
        self.trained = False
        self.model_path = None
    
    def load_and_train(self, csv_path: str, use_event_specific: bool = True):
        """Load data and train model"""
        print(f"📥 Loading data from {csv_path}...")
        profiles = self.data_pipeline.load_from_csv(csv_path)
        
        print(f"📊 Training on {len(profiles)} athlete profiles...")
        print(f"   Event categories: {set(p.event_category for p in profiles)}")
        
        metrics = self.predictor.train(profiles, use_event_specific=use_event_specific)
        
        print(f"✅ Training complete:")
        print(f"   MAE: {metrics['mae']:.3f}s")
        print(f"   R²: {metrics['r2']:.3f}")
        print(f"   Mean Uncertainty: ±{metrics['mean_uncertainty']:.3f}s")
        
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
        """Export certificate to JSON with proper serialization"""
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(certificate, f, default=convert_numpy, indent=2)
        
        print(f"📄 Report exported to {filepath}")
    
    def save_model(self, filepath: str):
        """Save trained model"""
        self.predictor.save_model(filepath)
        self.model_path = filepath
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        self.predictor = EnhancedSwimPredictor.load_model(filepath)
    
        # Reconnect report generator to loaded predictor
        self.report_generator = ComprehensiveReportGenerator(
            self.predictor,
            self.simulator
        )
    
        self.trained = True
        self.model_path = filepath
        print(f"📂 Model loaded from {filepath}")

# =============================================================================
# Example Usage with Enhanced Features
# =============================================================================

def create_enhanced_sample_data() -> str:
    """Create enhanced sample CSV with dynamic column prefixes"""
    import tempfile
    
    # Generate 50 sample athletes with realistic variation
    np.random.seed(42)
    n_samples = 50
    
    data = {
        'name': [f"Athlete_{i}" for i in range(n_samples)],
        'event': np.random.choice(
            ['50m Free', '100m Free', '200m Free', '400m Free', '200m IM'],
            n_samples
        ),
        'previous_best_time': np.random.normal(55, 5, n_samples),
        'target_time': np.random.normal(54, 5, n_samples),
        
        # Biometrics with bio_ prefix
        'bio_hrv': np.random.normal(65, 10, n_samples),
        'bio_resting_hr': np.random.normal(50, 5, n_samples),
        'bio_sleep_hours': np.random.normal(8, 1, n_samples),
        'bio_sleep_quality': np.random.normal(80, 10, n_samples),
        'bio_muscle_soreness': np.random.uniform(1, 10, n_samples),
        'bio_body_mass': np.random.normal(75, 8, n_samples),
        'bio_hydration_estimate': np.random.normal(95, 3, n_samples),
        'bio_blood_lactate': np.random.exponential(2, n_samples),
        
        # Environment with env_ prefix
        'env_water_temp': np.random.normal(27.5, 1, n_samples),
        'env_air_temp': np.random.normal(24, 2, n_samples),
        'env_humidity': np.random.normal(55, 10, n_samples),
        'env_pressure': np.random.normal(1013, 5, n_samples),
        'env_altitude': np.random.choice([0, 500, 1000], n_samples),
        'env_lane_number': np.random.randint(1, 9, n_samples),
        'env_pool_type': np.random.choice(['LCM', 'SCM', 'SCY'], n_samples),
        
        # Stroke metrics with stroke_ prefix
        'stroke_stroke_rate': np.random.normal(55, 8, n_samples),
        'stroke_stroke_length': np.random.normal(2.1, 0.3, n_samples),
        'stroke_distance_per_stroke': np.random.normal(2.0, 0.3, n_samples),
        'stroke_turn_time': np.random.normal(0.8, 0.2, n_samples),
        'stroke_underwater_distance': np.random.normal(12, 3, n_samples),
        'stroke_intra_cycle_velocity_fluctuation': np.random.exponential(0.15, n_samples),
        
        # Cognitive with cog_ prefix
        'cog_stress_level': np.random.uniform(1, 10, n_samples),
        'cog_screen_time': np.random.exponential(3, n_samples),
        'cog_gaming_hours': np.random.exponential(1, n_samples),
        'cog_focus_rating': np.random.uniform(1, 10, n_samples),
        'cog_pre_race_anxiety': np.random.uniform(1, 10, n_samples),
        
        # Training with train_ prefix
        'train_acute_load': np.random.normal(800, 200, n_samples),
        'train_chronic_load': np.random.normal(750, 150, n_samples),
        'train_weekly_yardage': np.random.normal(35000, 5000, n_samples),
        'train_training_stress_balance': np.random.normal(0, 3, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name


def main():
    """Enhanced example demonstrating all advanced features"""
    print("="*70)
    print("🚀 HYDROPREDICT ELITE - Advanced Swim Performance Engine v2.0")
    print("="*70)
    
    # Initialize app with ensemble
    print("\n⚙️ Initializing system with 5-model ensemble...")
    app = HydroPredictEliteApp(model_type='auto', n_ensemble=5)
    
    # Create enhanced sample data
    print("\n📝 Creating enhanced sample dataset...")
    sample_csv = create_enhanced_sample_data()
    
    # Load and train with event-specific models
    print("\n🎯 Training with event-specific modeling...")
    profiles = app.load_and_train(sample_csv, use_event_specific=True)
    
    # Save model
    app.save_model("hydropredict_elite_model.pkl")
    
    # Analyze a specific athlete
    print("\n🔬 Generating comprehensive performance certificate...")
    test_athlete = profiles[0]
    certificate = app.analyze_athlete(test_athlete)
    
    # Print certificate
    app.report_generator.print_certificate(certificate)
    
    # Export to JSON
    app.export_report(certificate, f"{test_athlete.name}_certificate.json")
    
    # Batch analysis example
    print("\n📊 Batch analysis for all athletes...")
    all_certificates = app.batch_analyze(profiles[:5])  # First 5 athletes
    
    # Show summary statistics
    predicted_times = [c['performance_metrics']['predicted_time'] for c in all_certificates]
    uncertainties = [c['model_confidence']['prediction_uncertainty'] for c in all_certificates]
    
    print(f"\n📈 Batch Summary:")
    print(f"   Athletes analyzed: {len(all_certificates)}")
    print(f"   Mean predicted time: {np.mean(predicted_times):.2f}s")
    print(f"   Mean uncertainty: ±{np.mean(uncertainties):.3f}s")
    
    # Demonstrate model loading
    print("\n🔄 Demonstrating model persistence...")
    new_app = HydroPredictEliteApp()
    new_app.load_model("hydropredict_elite_model.pkl")
    
    # Verify loaded model works
    test_certificate = new_app.analyze_athlete(test_athlete)
    print(f"✅ Loaded model prediction: {test_certificate['performance_metrics']['predicted_time']:.2f}s")
    
    # Cleanup
    os.unlink(sample_csv)
    
    print("\n" + "="*70)
    print("✅ HydroPredict Elite demonstration complete!")
    print("="*70)


if __name__ == "__main__":
    main()
