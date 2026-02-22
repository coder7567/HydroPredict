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
from sklearn.preprocessing import StandardScaler
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
    def __init__(self, n_estimators: int = 5, model_type: str = 'auto'):
        self.n_estimators = n_estimators
        self.models = []
        self.model_type = model_type
        self.is_fitted = False
        self.feature_names = None

    def feature_importance_seconds(
        self,
        X_val: np.ndarray,
        profiles_val: List[Any],   # SwimmerProfile list
        y_val_native: np.ndarray,  # pct/delta/absolute labels for those profiles
        target_type: str
    ) -> Dict[str, float]:
        """
        Permutation importance measured by seconds-space MAE, even if model was trained on pct/delta.
        """
        if not self.is_fitted or not self.models:
            raise ValueError("Ensemble must be fitted before computing importance")

        if self.feature_names is None:
            self.feature_names = [f"f{i}" for i in range(X_val.shape[1])]

        # Helper: convert native -> seconds for given profiles
        prev = np.asarray([p.previous_best_time for p in profiles_val], dtype=float)

        if len(prev) != len(y_val_native):
            raise ValueError(f"profiles_val length {len(prev)} != y_val_native length {len(y_val_native)}")

        def native_to_seconds(arr_native: np.ndarray) -> np.ndarray:
            if target_type in ("target_time", "absolute"):
                return arr_native
            if target_type == "delta":
                return prev + arr_native
            if target_type in ("pct", "pct_impr"):
                return prev * (1.0 + arr_native)
            return arr_native

        base_pred_native = self.models[0].predict(X_val)
        base_pred_sec = native_to_seconds(base_pred_native)
        true_sec = native_to_seconds(np.asarray(y_val_native, dtype=float))

        base_mae = np.mean(np.abs(base_pred_sec - true_sec))

        importances = []
        rng = np.random.RandomState(42)

        for j in range(X_val.shape[1]):
            Xp = X_val.copy()
            rng.shuffle(Xp[:, j])
            pred_native = self.models[0].predict(Xp)
            pred_sec = native_to_seconds(pred_native)
            mae = np.mean(np.abs(pred_sec - true_sec))
            importances.append(mae - base_mae)  # MAE increase = importance

        return dict(zip(self.feature_names, importances))

    def feature_importance(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Permutation importance using the first base model in the ensemble.
        Returns dict: feature_name -> importance
        """
        if not self.is_fitted or not self.models:
            raise ValueError("Ensemble must be fitted before computing importance")

        if self.feature_names is None:
            # fallback names if none provided
            self.feature_names = [f"f{i}" for i in range(X_val.shape[1])]

        r = permutation_importance(
            self.models[0],
            X_val,
            y_val,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        return dict(zip(self.feature_names, r.importances_mean))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.models = []
        rng = np.random.RandomState(42)
        n = X.shape[0]

        for i in range(self.n_estimators):
            idx = rng.randint(0, n, size=n)
            model = ModelFactory.create_model(self.model_type, random_state=42 + i)
            model.fit(X[idx], y[idx])
            self.models.append(model)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        all_preds = np.array([m.predict(X) for m in self.models])  # (n_models, n_samples)
        mean_pred = np.mean(all_preds, axis=0)
        unc = np.std(all_preds, axis=0)
        return mean_pred, unc

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
        df = pd.read_csv(filepath)

        ALIASES = {
            "env_pressure": "env_barometric_pressure",
            "bio_hydration": "bio_hydration_estimate",
        }
        df = df.rename(columns=ALIASES)

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
        
        # Core fields
        name = row.get('name', 'Unknown')
        event = row.get('event', 'Unknown')
        previous_best = float(row.get('previous_best_time', row.get('last_time', 0)))
        target = float(row.get('target_time', previous_best * 0.98))  # Default 2% improvement

        # ✅ OPTION B: keep times native. Only parse pool type.
        pool_raw = row.get('env_pool_type', None)
        if pool_raw is not None:
            try:
                pool_enum = PoolType(str(pool_raw).strip())
            except Exception:
                pool_enum = PoolType.LCM
        else:
            pool_enum = PoolType.LCM

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
            'pool_type': pool_enum  # ✅ keep actual pool type
        })

        # Convert pool_type string to Enum if needed
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

    def _convert_time_to_lcm(self, time_sec: float, pool_type: PoolType, event: str) -> float:
        """Approximate conversion of times in seconds from SCY/SCM to LCM-equivalent.

        This uses conservative, distance-aware multipliers to normalize times into
        a long-course-meter (LCM) scale for modeling consistency.
        """
        if pool_type == PoolType.LCM:
            return float(time_sec)

        # Base multipliers
        if pool_type == PoolType.SCM:
            mul = 1.02
        elif pool_type == PoolType.SCY:
            mul = 1.11
        else:
            mul = 1.0

        # Adjust slightly by event distance (sprints vs distance)
        try:
            import re
            m = re.search(r"(\d{2,4})", str(event))
            dist = int(m.group(1)) if m else None
        except Exception:
            dist = None

        if dist is not None:
            if dist <= 100:
                if pool_type == PoolType.SCY:
                    mul = 1.09
                elif pool_type == PoolType.SCM:
                    mul = 1.015
            elif dist >= 800:
                if pool_type == PoolType.SCY:
                    mul = 1.14
                elif pool_type == PoolType.SCM:
                    mul = 1.03

        return float(time_sec * mul)


class AdvancedFeaturePipeline:
    """Feature engineering with dynamic normalization and event grouping"""
    EVENT_CAT_MAP = {
        "SPRINT": 0,
        "MID_DISTANCE": 1,
        "DISTANCE": 2,
        "IM": 3,
        "OTHER": 4
    }
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.normalizers = {}  # For percentile-based normalization
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, profiles: List[SwimmerProfile]):
        # 1) compute normalizers first
        self._calculate_normalizers(profiles)

        # 2) build X USING those normalizers
        X = self._profiles_to_matrix(profiles, use_normalizers=True)

        self.feature_names = self._generate_feature_names()
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, profiles: Union[SwimmerProfile, List[SwimmerProfile]]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        if isinstance(profiles, SwimmerProfile):
            profiles = [profiles]

        X = self._profiles_to_matrix(profiles, use_normalizers=True)
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
    
    def _profiles_to_matrix(self, profiles: List[SwimmerProfile], use_normalizers: bool) -> np.ndarray:
        matrices = []
        bio_norm = self.normalizers.get('bio', {}) if use_normalizers else None
        stroke_norm = self.normalizers.get('stroke', {}) if use_normalizers else None

        for p in profiles:
            features = np.concatenate([
                [p.previous_best_time],
                p.biometrics.to_feature_vector(bio_norm),
                p.environment.to_feature_vector(),
                p.stroke.to_feature_vector(stroke_norm),
                p.cognitive.to_feature_vector(),
                p.training.to_feature_vector(),
                [
                    self.EVENT_CAT_MAP.get(p.event_category, 4)
                ]
            ])
            matrices.append(features)

        return np.array(matrices)
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for interpretability"""
        return [
            'previous_best_time',
            # Biometrics
            'hrv', 'resting_hr', 'sleep_hours', 'sleep_quality',
            'muscle_soreness', 'body_mass', 'hydration', 'lactate',
            # Environment
            'water_temp', 'air_temp', 'humidity', 'barometric_pressure',
            'altitude', 'lane_number', 'pool_type',
            # Stroke
            'stroke_rate', 'stroke_length', 'distance_per_stroke',
            'turn_time', 'underwater_distance', 'icvf',
            # Cognitive
            'stress_level', 'screen_time', 'gaming_hours', 'focus_rating', 'pre_race_anxiety',
            # Training
            'acute_load', 'chronic_load', 'acwr', 'weekly_yardage', 'training_stress_balance',
            # Encoded category
            'event_category_encoded'
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
        self.target_type = 'target_time'  # 'target_time' | 'delta' | 'pct'
        
        # Event-specific models
        self.event_models = {}
        # Pool-specific models (keys are PoolType)
        self.pool_models = {}
        self.pool_model_quality = {}  # PoolType -> dict(metrics + gate decision)
        self.use_pool_gating = True   # enable gating by default
        self.pool_gate_margin_pct = 0.02 # Only use pool-specific model if it shows at least this much improvement in MAE (percent) during CV

    def get_feature_importance(self) -> Dict[str, float]:
        """Return stored feature importance dict (may be empty if not computed)."""
        fi = getattr(self, "feature_importance", None)
        return fi if isinstance(fi, dict) else {}
        
    def train(self, profiles: List[SwimmerProfile], use_event_specific: bool = True, use_pool_specific: bool = True, target: str = 'target_time'):

        self.event_models = {}
        self.pool_models = {}
        self.pool_model_quality = {}
        # remember how we trained (absolute target, delta, or percent)
        self.target_type = target
        # Fit one shared preprocessing pipeline across all model variants
        self.feature_pipeline = AdvancedFeaturePipeline().fit(profiles)

        # (1) Train GENERAL FIRST so gating can compare against it
        general = self._train_single(profiles, target, pipeline=self.feature_pipeline)
        self.ensemble = general["ensemble"]
        self.feature_pipeline = general["pipeline"]
        self.training_history = general["training_history"]
        self.feature_importance = general["feature_importance"]

        # (2) Event-specific models (optional)
        if use_event_specific:
            event_groups = {}
            for p in profiles:
                event_groups.setdefault(p.event_category, []).append(p)

            for category, cat_profiles in event_groups.items():
                if len(cat_profiles) >= 10:
                    self.event_models[category] = self._train_single(
                        cat_profiles,
                        target,
                        pipeline=self.feature_pipeline
                    )
                else:
                    warnings.warn(f"Category {category} has only {len(cat_profiles)} samples, using general model")

        # (3) Pool-specific models WITH gating (optional)
        if use_pool_specific:
            pool_groups = {}
            for p in profiles:
                pool_groups.setdefault(p.environment.pool_type, []).append(p)

            for pool_type, pool_profiles in pool_groups.items():
                if len(pool_profiles) >= 10:
                    bundle, q = self._train_pool_model_with_gate(
                        pool_profiles,
                        target,
                        pipeline=self.feature_pipeline
                    )
                    self.pool_models[pool_type] = bundle
                    self.pool_model_quality[pool_type] = q
                else:
                    self.pool_model_quality[pool_type] = {
                        "n": len(pool_profiles),
                        "use_pool_model": False,
                        "reason": "too_few_samples"
                    }

        self.training_timestamp = datetime.now()
        return self.training_history
    
    def _train_single(
        self,
        profiles: List[SwimmerProfile],
        target: str,
        pipeline: Optional[AdvancedFeaturePipeline] = None
    ):
        # Use caller-provided shared pipeline when available
        if pipeline is None:
            pipeline = AdvancedFeaturePipeline().fit(profiles)
        X = pipeline.transform(profiles)

        # Build target according to requested target type
        if target == 'target_time' or target == 'absolute':
            y = np.array([p.target_time for p in profiles])
        elif target == 'delta':
            # target - previous_best_time (seconds)
            y = np.array([p.target_time - p.previous_best_time for p in profiles])
        elif target == 'pct' or target == 'pct_impr':
            # fractional improvement (target/previous - 1)
            y = np.array([(p.target_time / p.previous_best_time) - 1.0 for p in profiles])
        else:
            # fallback to absolute
            y = np.array([p.target_time for p in profiles])

        # Use K-Fold CV for more stable estimates when dataset is small
        kf = KFold(n_splits=min(5, max(2, len(profiles)//5)), shuffle=True, random_state=42)

        maes = []
        rmses = []
        r2s = []
        maes_seconds = []
        rmses_seconds = []
        r2s_seconds = []
        mean_uncs = []

        # Train ensemble per fold and collect metrics
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            ensemble = EnsembleModel(n_estimators=self.ensemble.n_estimators, model_type=self.ensemble.model_type)
            ensemble.feature_names = pipeline.feature_names
            ensemble.fit(X_train, y_train)

            y_pred, uncertainty = ensemble.predict(X_val)

            maes.append(mean_absolute_error(y_val, y_pred))
            rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2s.append(r2_score(y_val, y_pred))

            # Always keep comparable absolute-seconds metrics for reporting.
            val_profiles = [profiles[i] for i in val_idx]
            y_val_sec = self._native_to_absolute(y_val, val_profiles, target)
            y_pred_sec = self._native_to_absolute(y_pred, val_profiles, target)
            maes_seconds.append(mean_absolute_error(y_val_sec, y_pred_sec))
            rmses_seconds.append(np.sqrt(mean_squared_error(y_val_sec, y_pred_sec)))
            r2s_seconds.append(r2_score(y_val_sec, y_pred_sec))
            mean_uncs.append(float(np.mean(uncertainty)))

        # Aggregate CV metrics
        training_history = {
            'mae': float(np.mean(maes)),
            'rmse': float(np.mean(rmses)),
            'r2': float(np.mean(r2s)),
            'mae_seconds': float(np.mean(maes_seconds)),
            'rmse_seconds': float(np.mean(rmses_seconds)),
            'r2_seconds': float(np.mean(r2s_seconds)),
            'mean_uncertainty': float(np.mean(mean_uncs)),
            'n_samples': len(profiles),
            'cv_folds': kf.get_n_splits()
        }

        # Refit final model on full data for serving
        final_ensemble = EnsembleModel(n_estimators=self.ensemble.n_estimators, model_type=self.ensemble.model_type)
        final_ensemble.feature_names = pipeline.feature_names
        final_ensemble.fit(X, y)

        # Compute feature importance on full data using permutation on a held-out split if possible
        feature_importance = {}
        try:
            if len(profiles) >= 10:
                idx = np.arange(len(profiles))
                idx_tr, idx_va = train_test_split(idx, test_size=0.2, random_state=42)

                X_tr, X_va = X[idx_tr], X[idx_va]
                y_tr, y_va = y[idx_tr], y[idx_va]
                val_profiles = [profiles[i] for i in idx_va]  # ✅ aligned with X_va/y_va

                imp_ensemble = EnsembleModel(
                    n_estimators=self.ensemble.n_estimators,
                    model_type=self.ensemble.model_type
                )
                imp_ensemble.feature_names = pipeline.feature_names
                imp_ensemble.fit(X_tr, y_tr)

                feature_importance = imp_ensemble.feature_importance_seconds(
                    X_va, val_profiles, y_va, target
                )
            else:
                feature_importance = {fn: 0.0 for fn in pipeline.feature_names}
        except Exception as e:
            print("⚠️ Feature importance failed:", repr(e))
            feature_importance = {fn: 0.0 for fn in pipeline.feature_names}

        return {
            "ensemble": final_ensemble,
            "pipeline": pipeline,
            "training_history": training_history,
            "feature_importance": feature_importance
        }

    def _native_to_absolute(self, values: np.ndarray, profiles: List[SwimmerProfile], target: str) -> np.ndarray:
        """Convert native target-space values to absolute seconds for comparison/reporting."""
        arr = np.asarray(values, dtype=float)
        if target in ('target_time', 'absolute'):
            return arr

        prev = np.asarray([p.previous_best_time for p in profiles], dtype=float)
        if target == 'delta':
            return prev + arr
        if target in ('pct', 'pct_impr'):
            return prev * (1.0 + arr)
        return arr

    def _eval_bundle_mae(self, bundle, profiles: List[SwimmerProfile]) -> float:
        errs = []
        for p in profiles:
            pred, _ = self._predict_with_bundle(p, bundle)
            errs.append(abs(pred - p.target_time))
        return float(np.mean(errs)) if errs else float("inf")

    def _bundle_predict_native(self, bundle, profiles: List[SwimmerProfile]) -> np.ndarray:
        """Predict in the model's native target space (absolute/delta/pct)."""
        X = bundle["pipeline"].transform(profiles)
        pred, _ = bundle["ensemble"].predict(X)
        return np.asarray(pred, dtype=float)

    def _native_targets(self, profiles: List[SwimmerProfile], target: str) -> np.ndarray:
        """Build ground-truth vector in the same native target space as training."""
        if target in ('target_time', 'absolute'):
            return np.array([p.target_time for p in profiles], dtype=float)
        if target == 'delta':
            return np.array([p.target_time - p.previous_best_time for p in profiles], dtype=float)
        if target in ('pct', 'pct_impr'):
            return np.array([(p.target_time / p.previous_best_time) - 1.0 for p in profiles], dtype=float)
        return np.array([p.target_time for p in profiles], dtype=float)

    def _train_pool_model_with_gate(
        self,
        pool_profiles: List[SwimmerProfile],
        target: str,
        pipeline: Optional[AdvancedFeaturePipeline] = None
    ) -> Tuple[dict, dict]:
        """
        Train pool model as a RESIDUAL corrector on top of general model.
        Gate using seconds-MAE improvement.
        Returns: (pool_bundle, quality_dict)
        """
        # Too small: still train residual, but don't gate hard
        if len(pool_profiles) < 12:
            # Train residual on full pool_profiles
            X_pool = (pipeline or self.feature_pipeline).transform(pool_profiles)

            y_true_native = self._native_targets(pool_profiles, target)

            general_bundle = {"pipeline": self.feature_pipeline, "ensemble": self.ensemble}
            general_pred_native = self._bundle_predict_native(general_bundle, pool_profiles)

            y_resid = y_true_native - general_pred_native

            pool_ens = EnsembleModel(n_estimators=self.ensemble.n_estimators, model_type=self.ensemble.model_type)
            pool_ens.feature_names = (pipeline or self.feature_pipeline).feature_names
            pool_ens.fit(X_pool, y_resid)

            bundle = {
                "ensemble": pool_ens,
                "pipeline": (pipeline or self.feature_pipeline),
                "residual": True,
                "target_space": target
            }
            q = {
                "n": len(pool_profiles),
                "use_pool_model": True,
                "reason": "small_pool_no_holdout_residual"
            }
            return bundle, q

        train_p, val_p = train_test_split(pool_profiles, test_size=0.25, random_state=42)

        pipe = pipeline or self.feature_pipeline

        # ----- TRAIN residual model on train_p -----
        X_train = pipe.transform(train_p)

        y_train_true_native = self._native_targets(train_p, target)
        general_bundle = {"pipeline": self.feature_pipeline, "ensemble": self.ensemble}
        general_train_pred_native = self._bundle_predict_native(general_bundle, train_p)

        y_train_resid = y_train_true_native - general_train_pred_native

        pool_ens = EnsembleModel(n_estimators=self.ensemble.n_estimators, model_type=self.ensemble.model_type)
        pool_ens.feature_names = pipe.feature_names
        pool_ens.fit(X_train, y_train_resid)

        pool_bundle = {
            "ensemble": pool_ens,
            "pipeline": pipe,
            "residual": True,
            "target_space": target
        }

        # ----- EVAL on val_p (seconds MAE gating) -----
        # Native truths + general native preds
        y_val_true_native = self._native_targets(val_p, target)
        general_val_pred_native = self._bundle_predict_native(general_bundle, val_p)

        # Pool residual preds
        X_val = pipe.transform(val_p)
        resid_pred_native, _ = pool_ens.predict(X_val)

        pool_val_pred_native = general_val_pred_native + resid_pred_native

        # Convert to seconds for gating
        y_val_true_sec = self._native_to_absolute(y_val_true_native, val_p, target)
        gen_val_pred_sec = self._native_to_absolute(general_val_pred_native, val_p, target)
        pool_val_pred_sec = self._native_to_absolute(pool_val_pred_native, val_p, target)

        general_mae_seconds = float(np.mean(np.abs(gen_val_pred_sec - y_val_true_sec)))
        pool_mae_seconds = float(np.mean(np.abs(pool_val_pred_sec - y_val_true_sec)))

        # Gate: use pool model if it is not worse (or better by margin)
        use_pool = pool_mae_seconds <= (general_mae_seconds * (1.0 - self.pool_gate_margin_pct))

        q = {
            "n": len(pool_profiles),
            "target_space": target,
            "pool_mae_seconds": pool_mae_seconds,
            "general_mae_seconds": general_mae_seconds,
            "delta_mae_seconds": (general_mae_seconds - pool_mae_seconds),
            "pool_gate_margin_pct": float(self.pool_gate_margin_pct),
            "use_pool_model": bool(use_pool),
            "reason": "pool_better_or_equal_seconds" if use_pool else "pool_worse_than_general_seconds"
        }

        # keep compatibility fields (optional)
        q["pool_mae"] = q["pool_mae_seconds"]
        q["general_mae"] = q["general_mae_seconds"]
        q["delta_mae"] = q["delta_mae_seconds"]

        return pool_bundle, q
    
    def predict(self, profile: SwimmerProfile) -> Tuple[float, float]:
        """
        Default prediction API used by evaluate() and report generator.
        Uses pool-routing when available (with gating), otherwise falls back to general model.
        Returns (predicted_time_seconds, uncertainty_seconds).
        """
        pred, unc, _used_pool = self.predict_pool_routed(profile)
        return pred, unc

    def predict_general_only(self, profile: SwimmerProfile) -> Tuple[float, float]:
        bundle = {"pipeline": self.feature_pipeline, "ensemble": self.ensemble}
        return self._predict_with_bundle(profile, bundle)
    
    def _predict_with_bundle(self, profile: SwimmerProfile, bundle) -> Tuple[float, float]:
        X = bundle["pipeline"].transform(profile)
        pred, unc = bundle["ensemble"].predict(X)
        pred_val = float(pred[0])
        unc_val = float(unc[0])

        tt = getattr(self, 'target_type', 'target_time')
        if tt == 'delta':
            return profile.previous_best_time + pred_val, unc_val
        elif tt in ('pct', 'pct_impr'):
            return profile.previous_best_time * (1.0 + pred_val), unc_val * profile.previous_best_time
        else:
            return pred_val, unc_val
        
    def predict_pool_routed(self, profile: SwimmerProfile) -> Tuple[float, float, bool]:
        bundle = self.pool_models.get(profile.environment.pool_type)
        pool_ok = True
        q = getattr(self, "pool_model_quality", {}).get(profile.environment.pool_type)
        if q is not None and getattr(self, "use_pool_gating", True):
            pool_ok = bool(q.get("use_pool_model", True))

        used_pool_model = (bundle is not None and pool_ok)

        if used_pool_model:
            general_bundle = {"pipeline": self.feature_pipeline, "ensemble": self.ensemble}
            gen_native = self._bundle_predict_native(general_bundle, [profile])[0]

            Xp = bundle["pipeline"].transform(profile)
            resid_native, resid_unc = bundle["ensemble"].predict(Xp)
            pred_native = float(gen_native + resid_native[0])
            unc_native = float(resid_unc[0])
        else:
            # fallback to general native
            general_bundle = {"pipeline": self.feature_pipeline, "ensemble": self.ensemble}
            Xg = general_bundle["pipeline"].transform(profile)
            pred_native_arr, unc_native_arr = general_bundle["ensemble"].predict(Xg)
            pred_native = float(pred_native_arr[0])
            unc_native = float(unc_native_arr[0])

        # Convert native -> absolute seconds same way as _predict_with_bundle does
        tt = getattr(self, 'target_type', 'target_time')
        if tt == 'delta':
            return profile.previous_best_time + pred_native, unc_native, used_pool_model
        elif tt in ('pct', 'pct_impr'):
            return profile.previous_best_time * (1.0 + pred_native), unc_native * profile.previous_best_time, used_pool_model
        else:
            return pred_native, unc_native, used_pool_model

    def predict_event_routed(self, profile: SwimmerProfile) -> Tuple[float, float]:
        """Force event routing: event model if exists else general."""
        bundle = self.event_models.get(profile.event_category)
        if bundle is None:
            bundle = {"pipeline": self.feature_pipeline, "ensemble": self.ensemble}
        return self._predict_with_bundle(profile, bundle)
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'ensemble': self.ensemble,
            'feature_pipeline': self.feature_pipeline,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version,
            'training_timestamp': self.training_timestamp,
            'event_models': self.event_models,
            'pool_models': self.pool_models,
            'pool_model_quality': self.pool_model_quality,
            'use_pool_gating': self.use_pool_gating,
            'pool_gate_margin_pct': self.pool_gate_margin_pct,
            'target_type': getattr(self, 'target_type', 'target_time')
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
        predictor.pool_models = model_data.get('pool_models', {})
        predictor.pool_model_quality = model_data.get('pool_model_quality', {})
        predictor.use_pool_gating = model_data.get('use_pool_gating', True)
        predictor.pool_gate_margin_pct = model_data.get('pool_gate_margin_pct', 0.10)
        predictor.target_type = model_data.get('target_type', 'target_time')
        
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
        feature_importance = self.predictor.feature_importance if isinstance(getattr(self.predictor, "feature_importance", None), dict) else {}
        
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

        target_type = getattr(self.predictor, 'target_type', 'target_time')
        model_mae_seconds = self.predictor.training_history.get(
            'mae_seconds',
            self.predictor.training_history.get('mae')
        )
        model_r2_seconds = self.predictor.training_history.get(
            'r2_seconds',
            self.predictor.training_history.get('r2')
        )
        
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
                    'absolute_error': abs(pred_time - profile.target_time),
                    'relative_error': (abs(pred_time - profile.target_time) / profile.target_time) if profile.target_time != 0 else None,
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
                # Always report seconds-space metrics for comparability.
                'model_mae': model_mae_seconds,
                'model_r2': model_r2_seconds,
                'feature_importance': dict(sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:15])
            }
        }

        if target_type not in ['target_time', 'absolute']:
            certificate['model_confidence']['model_mae_native'] = self.predictor.training_history.get('mae')
            certificate['model_confidence']['model_r2_native'] = self.predictor.training_history.get('r2')
        
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
        if certificate['performance_metrics'].get('absolute_error') is not None:
            print(f"   Absolute Error: {certificate['performance_metrics']['absolute_error']:.2f}s")
        if certificate['performance_metrics'].get('relative_error') is not None:
            print(f"   Relative Error: {certificate['performance_metrics']['relative_error']*100:.2f}%")
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
        if certificate['model_confidence']['model_mae'] is not None:
            print(f"   Model MAE (seconds): {certificate['model_confidence']['model_mae']:.3f}s")
            print(f"   Model R² (seconds): {certificate['model_confidence']['model_r2']:.3f}")
            if certificate['model_confidence'].get('model_mae_native') is not None:
                print(f"   Model MAE (native target): {certificate['model_confidence']['model_mae_native']:.6f}")
            if certificate['model_confidence'].get('model_r2_native') is not None:
                print(f"   Model R² (native target): {certificate['model_confidence']['model_r2_native']:.6f}")
        else:
            print(f"   Model MAE / R²: N/A")
        
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
    
    def choose_athlete(self, profiles):
        """
        Interactive selector for athlete profiles.
        Returns a SwimmerProfile.
        """
        if not profiles:
            raise ValueError("No profiles loaded.")

        # Build a display list with indexes
        print("\n🧾 Available athletes:")
        for i, p in enumerate(profiles, 1):
            pool = getattr(p.environment.pool_type, "value", str(p.environment.pool_type))
            print(f"  {i:3d}) {p.name:25s} | {p.event:12s} | {pool}")

        while True:
            raw = input("\nPick an athlete by number OR type part of a name: ").strip()

            # If number selection
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(profiles):
                    return profiles[idx - 1]
                print(f"❌ Number out of range (1..{len(profiles)})")
                continue

            # Name search selection (partial match)
            q = raw.lower()
            matches = [p for p in profiles if q in p.name.lower()]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                print("\n🔎 Multiple matches:")
                for i, p in enumerate(matches, 1):
                    pool = getattr(p.environment.pool_type, "value", str(p.environment.pool_type))
                    print(f"  {i:2d}) {p.name:25s} | {p.event:12s} | {pool}")
                pick = input("Pick match number: ").strip()
                if pick.isdigit() and 1 <= int(pick) <= len(matches):
                    return matches[int(pick) - 1]
                print("❌ Invalid match pick.")
            else:
                print("❌ No matches. Try again.")
    
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

def main():
    """Enhanced example demonstrating all advanced features"""
    print("="*70)
    print("🚀 HYDROPREDICT ELITE - Advanced Swim Performance Engine v2.0")
    print("="*70)
    
    # Initialize app with ensemble
    print("\n⚙️ Initializing system with 5-model ensemble...")
    app = HydroPredictEliteApp(model_type='auto', n_ensemble=5)
    
    # Use provided swimmers_import.csv from repository instead of generating
    print("\n📝 Using provided swimmers_import.csv in repository...")
    generated_sample = False
    sample_csv = r"C:/Users/levir/Downloads/HydroPredict-main/swimmers_import.csv"
    profiles = app.data_pipeline.load_from_csv(sample_csv)

    # Holdout split for realistic evaluation (train on train, evaluate on test)
    pool_labels = [p.environment.pool_type.value for p in profiles]
    train_profiles, test_profiles = train_test_split(
        profiles,
        test_size=0.25,
        random_state=42,
        stratify=pool_labels
    )
    print(f"📚 Data split: train={len(train_profiles)} | test={len(test_profiles)}")

    # Experiment targets to try
    targets = ['target_time', 'delta', 'pct']
    results = {}

    def evaluate(predictor, profiles):
        # returns overall MAE, per-event MAE dict, per-pool MAE dict,
        # plus mean/median relative errors and per-event/pool relative MAE
        from collections import defaultdict
        import numpy as _np

        errors = []
        rel_errors = []
        per_event = defaultdict(list)
        per_event_rel = defaultdict(list)
        per_pool = defaultdict(list)
        per_pool_rel = defaultdict(list)

        for p in profiles:
            pred, _ = predictor.predict(p)
            true = p.target_time
            err = abs(pred - true)
            rel = abs(pred - true) / true if true != 0 else 0.0
            errors.append(err)
            rel_errors.append(rel)
            per_event[p.event].append(err)
            per_event_rel[p.event].append(rel)
            per_pool[p.environment.pool_type.value].append(err)
            per_pool_rel[p.environment.pool_type.value].append(rel)

        overall_mae = float(_np.mean(errors))
        overall_mean_rel = float(_np.mean(rel_errors))
        overall_median_rel = float(_np.median(rel_errors))

        per_event_mae = {k: float(_np.mean(v)) for k, v in per_event.items()}
        per_pool_mae = {k: float(_np.mean(v)) for k, v in per_pool.items()}

        per_event_rel_mae = {k: float(_np.mean(v)) for k, v in per_event_rel.items()}
        per_pool_rel_mae = {k: float(_np.mean(v)) for k, v in per_pool_rel.items()}

        return {
            'overall_mae': overall_mae,
            'overall_mean_rel': overall_mean_rel,
            'overall_median_rel': overall_median_rel,
            'per_event_mae': per_event_mae,
            'per_pool_mae': per_pool_mae,
            'per_event_rel_mae': per_event_rel_mae,
            'per_pool_rel_mae': per_pool_rel_mae
        }

    def per_pool_comparison_report(predictor: EnhancedSwimPredictor, profiles: List[SwimmerProfile], min_slice_n: int = 5):
        from collections import defaultdict

        def baseline_pred(p):
            return p.previous_best_time

        # Accumulators
        pool_rows = defaultdict(list)  # pool -> list of dicts per swimmer
        slice_rows = defaultdict(list) # (pool, event) -> list of dicts

        total = 0
        routed_used = 0

        for p in profiles:
            total += 1

            base = baseline_pred(p)
            gen, _ = predictor.predict_general_only(p)
            prt, _, used_pool = predictor.predict_pool_routed(p)

            if used_pool:
                routed_used += 1

            true = p.target_time

            def err(x):
                return abs(x - true)

            def rel(x):
                return (abs(x - true) / true) if true != 0 else 0.0

            row = {
                "base_err": err(base), "base_rel": rel(base),
                "gen_err":  err(gen),  "gen_rel":  rel(gen),
                "prt_err":  err(prt),  "prt_rel":  rel(prt),
            }

            pool_key = p.environment.pool_type.value
            pool_rows[pool_key].append(row)
            slice_rows[(pool_key, p.event)].append(row)

        def summarize(rows):
            if not rows:
                return None
            base_mae = float(np.mean([r["base_err"] for r in rows]))
            gen_mae = float(np.mean([r["gen_err"] for r in rows]))
            prt_mae = float(np.mean([r["prt_err"] for r in rows]))
            base_rel = float(np.mean([r["base_rel"] for r in rows]))
            gen_rel = float(np.mean([r["gen_rel"] for r in rows]))
            prt_rel = float(np.mean([r["prt_rel"] for r in rows]))
            return base_mae, base_rel, gen_mae, gen_rel, prt_mae, prt_rel

        overall = summarize([r for rows in pool_rows.values() for r in rows])

        print("\n" + "="*72)
        print("📊 PER-POOL COMPARISON REPORT (Baseline vs General vs Pool-Routed)")
        print("="*72)
        if overall:
            base_mae, base_rel, gen_mae, gen_rel, prt_mae, prt_rel = overall
            print("\nOverall:")
            print(f"  Baseline MAE: {base_mae:.3f}s | mean rel {base_rel*100:.2f}%")
            print(f"  General  MAE: {gen_mae:.3f}s | mean rel {gen_rel*100:.2f}%")
            print(f"  PoolRt   MAE: {prt_mae:.3f}s | mean rel {prt_rel*100:.2f}%")
            print(f"  Routing: pool model used {100.0*routed_used/total:.1f}% ({routed_used} / {total})")

        print("\nPer-pool:")
        for pool_key in sorted(pool_rows.keys()):
            rows = pool_rows[pool_key]
            base_mae, base_rel, gen_mae, gen_rel, prt_mae, prt_rel = summarize(rows)
            n = len(rows)
            print(f"\n  {pool_key} (n={n}):")
            print(f"    Baseline: MAE {base_mae:.3f}s | mean rel {base_rel*100:.2f}%")
            print(f"    General : MAE {gen_mae:.3f}s | mean rel {gen_rel*100:.2f}%   (Δ vs base {gen_mae-base_mae:+.3f}s)")
            print(f"    PoolRt  : MAE {prt_mae:.3f}s | mean rel {prt_rel*100:.2f}%   (Δ vs gen  {prt_mae-gen_mae:+.3f}s)")

        # Slice analysis (pool|event)
        helps = []
        hurts = []
        for (pool_key, event), rows in slice_rows.items():
            if len(rows) < min_slice_n:
                continue
            _, _, gen_mae, _, prt_mae, _ = summarize(rows)
            diff = gen_mae - prt_mae  # positive = PoolRt helps
            item = (diff, pool_key, event, len(rows), gen_mae, prt_mae)
            if diff >= 0:
                helps.append(item)
            else:
                hurts.append(item)

        helps.sort(reverse=True, key=lambda x: x[0])
        hurts.sort(key=lambda x: x[0])

        print(f"\nTop pool|event slices (n>={min_slice_n}) where PoolRt helps most vs General:")
        for diff, pool_key, event, n, gen_mae, prt_mae in helps[:10]:
            print(f"  +{diff:.3f}s  {pool_key} | {event} (n={n})  General {gen_mae:.3f}s -> PoolRt {prt_mae:.3f}s")

        print(f"\nWorst pool|event slices (n>={min_slice_n}) where PoolRt is worse than General:")
        for diff, pool_key, event, n, gen_mae, prt_mae in hurts[:10]:
            print(f"  {diff:.3f}s  {pool_key} | {event} (n={n})  General {gen_mae:.3f}s -> PoolRt {prt_mae:.3f}s")

        print("="*72)


    # Run experiments
    for t in targets:
        print(f"\n🔬 Training target: {t}")
        # create fresh app/predictor for each run
        exp_app = HydroPredictEliteApp(model_type='auto', n_ensemble=5)
        exp_app.data_pipeline = app.data_pipeline
        # train with chosen target
        exp_app.predictor.train(
            train_profiles,
            use_event_specific=True,
            use_pool_specific=True,
            target=t
        )
        print("Pool gate decisions:")
        for k, v in exp_app.predictor.pool_model_quality.items():
            print(" ", k.value, v)
        print("Pool models trained:", [k.value for k in exp_app.predictor.pool_models.keys()])
        eval_res = evaluate(exp_app.predictor, test_profiles)
        results[t] = {
            'overall_mae': eval_res['overall_mae'],
            'overall_mean_rel': eval_res['overall_mean_rel'],
            'overall_median_rel': eval_res['overall_median_rel'],
            'per_event_mae': eval_res['per_event_mae'],
            'per_pool_mae': eval_res['per_pool_mae'],
            'per_event_rel_mae': eval_res['per_event_rel_mae'],
            'per_pool_rel_mae': eval_res['per_pool_rel_mae'],
            'predictor': exp_app.predictor
        }
        print(f"   Completed {t} (holdout): overall MAE={eval_res['overall_mae']:.3f}s, mean rel={eval_res['overall_mean_rel']*100:.2f}%")

    # Compare per-event and per-pool averages
    def avg_dict(d):
        import numpy as _np
        return float(_np.mean(list(d.values()))) if d else float('inf')

    # Weighted winner selection: 70% overall MAE + 30% average per-event MAE
    scored = []
    for target_name, target_res in results.items():
        overall_mae = target_res['overall_mae']
        per_event_avg = avg_dict(target_res['per_event_mae'])
        score = overall_mae * 0.7 + per_event_avg * 0.3
        scored.append((score, overall_mae, target_name, per_event_avg))

    scored.sort(key=lambda x: (x[0], x[1]))  # tie-break on overall MAE
    winner = scored[0][2]

    print("\n🎯 Target scoring (0.7*overall + 0.3*per-event-avg):")
    for score, overall_mae, target_name, per_event_avg in scored:
        print(f"   {target_name:11s} score={score:.4f} | overall={overall_mae:.4f} | per-event-avg={per_event_avg:.4f}")

    print(f"\n🏆 Selected winning target: {winner}")

    # Use winner predictor as main app predictor and save
    app.predictor = results[winner]['predictor']
    app.feature_pipeline = app.predictor.feature_pipeline
    app.training_timestamp = app.predictor.training_timestamp
    # reconnect report generator and mark trained
    app.report_generator = ComprehensiveReportGenerator(app.predictor, app.simulator)
    app.trained = True

    per_pool_comparison_report(app.predictor, test_profiles, min_slice_n=5)

    app.save_model("hydropredict_elite_model.pkl")
    
    # Analyze a specific athlete
    print("\n🔬 Generating comprehensive performance certificate...")
    test_athlete = app.choose_athlete(profiles)
    certificate = app.analyze_athlete(test_athlete)
    
    # Print certificate
    app.report_generator.print_certificate(certificate)
    
    # Export to JSON
    app.export_report(certificate, f"{test_athlete.name}_certificate.json")
    
    # Batch analysis example
    print("\n📊 Batch analysis for all athletes...")
    all_certificates = app.batch_analyze(test_profiles[:5])  # First 5 holdout athletes
    
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
    
    # Cleanup (only remove if we generated a temporary sample file)
    if 'generated_sample' in locals() and generated_sample:
        os.unlink(sample_csv)
    
    print("\n" + "="*70)
    print("✅ HydroPredict Elite demonstration complete!")
    print("="*70)


if __name__ == "__main__":
    main()
