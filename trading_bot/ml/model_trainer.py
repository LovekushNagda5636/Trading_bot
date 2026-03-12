"""
Model Trainer for continuous learning and model improvement.
Handles training, validation, and deployment of ML models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Callable
import structlog
from dataclasses import dataclass, field
import pickle
import json
from pathlib import Path
import asyncio

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from ..core.models import TradingSignal, TradeDecision, Position
from .feature_engineering import FeatureEngineer, MarketFeatures

logger = structlog.get_logger(__name__)


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    cross_val_score: float = 0.0
    training_samples: int = 0
    validation_samples: int = 0
    training_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingData:
    """Training data point."""
    features: np.ndarray
    target: int  # 0: unprofitable, 1: profitable
    symbol: str
    timestamp: datetime
    trade_outcome: Dict[str, Any]
    market_conditions: Dict[str, Any]


class ModelTrainer:
    """
    Continuous learning system that trains and improves ML models.
    Handles data collection, model training, validation, and deployment.
    """
    
    def __init__(self, feature_engineer: FeatureEngineer):
        self.feature_engineer = feature_engineer
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'neural_network': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Trained models
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.best_model_name: Optional[str] = None
        
        # Training data
        self.training_data: List[TrainingData] = []
        self.max_training_data = 10000
        
        # Training configuration
        self.min_training_samples = 100
        self.retrain_threshold = 50  # Retrain after N new samples
        self.validation_split = 0.2
        self.cross_validation_folds = 5
        
        # Training state
        self.is_training = False
        self.last_training_time: Optional[datetime] = None
        self.training_counter = 0
        
        # Model persistence
        self.model_save_dir = Path("models")
        self.model_save_dir.mkdir(exist_ok=True)
        
        logger.info("Model Trainer initialized")
    
    def add_training_sample(
        self, 
        symbol: str, 
        features: MarketFeatures, 
        trade_decision: TradeDecision,
        trade_outcome: Dict[str, Any]
    ) -> None:
        """Add a new training sample."""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                return
            
            # Determine target (profitable or not)
            pnl = trade_outcome.get('pnl', 0)
            target = 1 if pnl > 0 else 0
            
            # Create training data point
            training_sample = TrainingData(
                features=feature_vector,
                target=target,
                symbol=symbol,
                timestamp=datetime.now(),
                trade_outcome=trade_outcome,
                market_conditions=self._extract_market_conditions(features)
            )
            
            self.training_data.append(training_sample)
            
            # Limit training data size
            if len(self.training_data) > self.max_training_data:
                self.training_data = self.training_data[-self.max_training_data:]
            
            self.training_counter += 1
            
            # Trigger retraining if threshold reached
            if (self.training_counter >= self.retrain_threshold and 
                len(self.training_data) >= self.min_training_samples):
                asyncio.create_task(self.retrain_models())
            
            logger.debug(f"Added training sample for {symbol}. Total samples: {len(self.training_data)}")
            
        except Exception as e:
            logger.error(f"Error adding training sample: {e}")
    
    def _prepare_feature_vector(self, features: MarketFeatures) -> Optional[np.ndarray]:
        """Prepare feature vector from market features."""
        try:
            feature_vector = []
            
            # Price features
            price_features = [
                features.price_features.get('return_5d', 0),
                features.price_features.get('return_20d', 0),
                features.price_features.get('momentum_10d', 0),
                features.price_features.get('price_vs_ma_20', 0),
                features.price_features.get('price_vs_ma_50', 0),
                features.price_features.get('price_percentile_50d', 0.5),
            ]
            feature_vector.extend(price_features)
            
            # Technical features
            technical_features = [
                features.technical_features.get('rsi', 50) / 100.0,
                features.technical_features.get('macd_bullish', 0),
                features.technical_features.get('bb_position', 0.5),
                features.technical_features.get('bb_squeeze', 0.02),
                features.technical_features.get('stoch_k', 50) / 100.0,
                features.technical_features.get('williams_r', -50) / -100.0,
            ]
            feature_vector.extend(technical_features)
            
            # Volume features
            volume_features = [
                features.volume_features.get('volume_ratio_5', 1.0),
                features.volume_features.get('volume_ratio_20', 1.0),
                features.volume_features.get('volume_trend', 0),
                features.volume_features.get('price_vs_vwap', 0),
            ]
            feature_vector.extend(volume_features)
            
            # Volatility features
            volatility_features = [
                features.volatility_features.get('volatility_10d', 0.2),
                features.volatility_features.get('volatility_20d', 0.2),
                features.volatility_features.get('volatility_50d', 0.2),
                features.volatility_features.get('vol_clustering', 1.0),
            ]
            feature_vector.extend(volatility_features)
            
            # Microstructure features
            microstructure_features = [
                features.microstructure_features.get('avg_spread', 0.001),
                features.microstructure_features.get('order_imbalance', 0),
                features.microstructure_features.get('price_impact', 0),
            ]
            feature_vector.extend(microstructure_features)
            
            # Temporal features
            temporal_features = [
                features.temporal_features.get('hour', 12) / 24.0,
                features.temporal_features.get('is_market_open', 0),
                features.temporal_features.get('is_opening_hour', 0),
                features.temporal_features.get('is_closing_hour', 0),
                features.temporal_features.get('is_friday', 0),
                features.temporal_features.get('sin_hour', 0),
                features.temporal_features.get('cos_hour', 1),
            ]
            feature_vector.extend(temporal_features)
            
            # Regime features
            regime_features = [
                features.regime_features.get('trend_regime', 0),
                features.regime_features.get('vol_regime', 0),
                features.regime_features.get('mean_reversion_regime', 0),
                features.regime_features.get('momentum_regime', 0),
            ]
            feature_vector.extend(regime_features)
            
            # Cross-asset features (placeholders)
            cross_asset_features = [
                features.cross_asset_features.get('market_correlation', 0.5),
                features.cross_asset_features.get('sector_correlation', 0.6),
                features.cross_asset_features.get('index_correlation', 0.7),
            ]
            feature_vector.extend(cross_asset_features)
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error preparing feature vector: {e}")
            return None
    
    def _extract_market_conditions(self, features: MarketFeatures) -> Dict[str, Any]:
        """Extract market conditions for analysis."""
        return {
            'volatility': features.volatility_features.get('volatility_20d', 0.2),
            'trend_strength': features.regime_features.get('trend_regime', 0),
            'volume_ratio': features.volume_features.get('volume_ratio_20', 1.0),
            'rsi': features.technical_features.get('rsi', 50),
            'bb_position': features.technical_features.get('bb_position', 0.5),
            'market_open': features.temporal_features.get('is_market_open', 0)
        }
    
    async def retrain_models(self) -> None:
        """Retrain all models with new data."""
        if self.is_training:
            logger.info("Training already in progress, skipping...")
            return
        
        try:
            self.is_training = True
            logger.info(f"Starting model retraining with {len(self.training_data)} samples...")
            
            # Prepare training data
            X, y = self._prepare_training_arrays()
            if X is None or len(X) < self.min_training_samples:
                logger.warning(f"Insufficient training data: {len(X) if X is not None else 0}")
                return
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split, random_state=42, stratify=y
            )
            
            # Train each model type
            for model_name, config in self.model_configs.items():
                await self._train_single_model(model_name, config, X_train, X_val, y_train, y_val)
            
            # Select best model
            self._select_best_model()
            
            # Save models
            self._save_models()
            
            self.last_training_time = datetime.now()
            self.training_counter = 0
            
            logger.info(f"Model retraining completed. Best model: {self.best_model_name}")
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
        finally:
            self.is_training = False
    
    def _prepare_training_arrays(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training arrays from training data."""
        try:
            if not self.training_data:
                return None, None
            
            X = np.array([sample.features for sample in self.training_data])
            y = np.array([sample.target for sample in self.training_data])
            
            # Remove any NaN or infinite values
            valid_indices = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) == 0:
                return None, None
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training arrays: {e}")
            return None, None
    
    async def _train_single_model(
        self, 
        model_name: str, 
        config: Dict, 
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        y_train: np.ndarray, 
        y_val: np.ndarray
    ) -> None:
        """Train a single model with hyperparameter optimization."""
        try:
            start_time = datetime.now()
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Hyperparameter optimization
            model_class = config['model']
            param_grid = config['params']
            
            # Use a subset of parameter combinations for faster training
            if len(X_train) < 1000:
                # Reduce parameter grid for small datasets
                param_grid = {k: v[:2] for k, v in param_grid.items()}
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_class(random_state=42),
                param_grid,
                cv=min(self.cross_validation_folds, len(X_train) // 20),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            
            # Validate model
            y_pred = best_model.predict(X_val_scaled)
            y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
            
            # Calculate performance metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            performance = ModelPerformance(
                accuracy=accuracy_score(y_val, y_pred),
                precision=precision_score(y_val, y_pred, zero_division=0),
                recall=recall_score(y_val, y_pred, zero_division=0),
                f1_score=f1_score(y_val, y_pred, zero_division=0),
                auc_score=roc_auc_score(y_val, y_pred_proba),
                cross_val_score=grid_search.best_score_,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                training_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Store model and performance
            self.models[model_name] = best_model
            self.scalers[model_name] = scaler
            self.model_performance[model_name] = performance
            
            logger.info(f"Trained {model_name}: "
                       f"Accuracy={performance.accuracy:.3f}, "
                       f"AUC={performance.auc_score:.3f}, "
                       f"CV Score={performance.cross_val_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
    
    # Minimum performance to deploy a model
    MIN_AUC_THRESHOLD = 0.52  # Must be better than random (0.50)
    MIN_IMPROVEMENT_PCT = 0.01  # 1% improvement required to replace

    def _select_best_model(self) -> None:
        """Select the best performing model — only if it clears validation."""
        try:
            if not self.model_performance:
                return
            
            # Compute previous best score
            prev_best_score = -1.0
            if self.best_model_name and self.best_model_name in self.model_performance:
                prev_perf = self.model_performance[self.best_model_name]
                prev_best_score = (
                    prev_perf.auc_score * 0.4 +
                    prev_perf.f1_score * 0.3 +
                    prev_perf.cross_val_score * 0.2 +
                    prev_perf.accuracy * 0.1
                )
            
            # Score all candidate models
            best_score = -1
            best_model = None
            
            for model_name, performance in self.model_performance.items():
                score = (
                    performance.auc_score * 0.4 +
                    performance.f1_score * 0.3 +
                    performance.cross_val_score * 0.2 +
                    performance.accuracy * 0.1
                )
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            # Gate 1: Minimum AUC threshold
            if best_model and self.model_performance[best_model].auc_score < self.MIN_AUC_THRESHOLD:
                logger.warning(
                    f"Model {best_model} AUC ({self.model_performance[best_model].auc_score:.3f}) "
                    f"below minimum {self.MIN_AUC_THRESHOLD}. NOT deploying."
                )
                return
            
            # Gate 2: Must improve over existing best
            if prev_best_score > 0 and best_score < prev_best_score * (1 + self.MIN_IMPROVEMENT_PCT):
                logger.info(
                    f"New best ({best_model}: {best_score:.3f}) does not improve "
                    f"over current ({self.best_model_name}: {prev_best_score:.3f}) "
                    f"by at least {self.MIN_IMPROVEMENT_PCT*100:.0f}%. Keeping current model."
                )
                return
            
            self.best_model_name = best_model
            logger.info(f"Selected best model: {best_model} (score: {best_score:.3f})")
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}")
    
    def predict(self, features: MarketFeatures) -> Optional[Tuple[int, float]]:
        """Make prediction using the best model."""
        try:
            if not self.best_model_name or self.best_model_name not in self.models:
                return None
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                return None
            
            # Scale features
            scaler = self.scalers[self.best_model_name]
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Make prediction
            model = self.models[self.best_model_name]
            prediction = model.predict(feature_vector_scaled)[0]
            confidence = model.predict_proba(feature_vector_scaled)[0].max()
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for model_name in self.models:
                # Save model
                model_file = self.model_save_dir / f"{model_name}_{timestamp}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(self.models[model_name], f)
                
                # Save scaler
                scaler_file = self.model_save_dir / f"{model_name}_scaler_{timestamp}.pkl"
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scalers[model_name], f)
            
            # Save performance metrics
            performance_file = self.model_save_dir / f"performance_{timestamp}.json"
            performance_data = {}
            for model_name, perf in self.model_performance.items():
                performance_data[model_name] = {
                    'accuracy': perf.accuracy,
                    'precision': perf.precision,
                    'recall': perf.recall,
                    'f1_score': perf.f1_score,
                    'auc_score': perf.auc_score,
                    'cross_val_score': perf.cross_val_score,
                    'training_samples': perf.training_samples,
                    'validation_samples': perf.validation_samples,
                    'training_time': perf.training_time
                }
            
            with open(performance_file, 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            # Save metadata
            metadata = {
                'best_model': self.best_model_name,
                'training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'total_samples': len(self.training_data),
                'feature_count': len(self.training_data[0].features) if self.training_data else 0
            }
            
            metadata_file = self.model_save_dir / f"metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Models saved with timestamp: {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, timestamp: str) -> bool:
        """Load models from disk."""
        try:
            for model_name in self.model_configs.keys():
                # Load model
                model_file = self.model_save_dir / f"{model_name}_{timestamp}.pkl"
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                
                # Load scaler
                scaler_file = self.model_save_dir / f"{model_name}_scaler_{timestamp}.pkl"
                if scaler_file.exists():
                    with open(scaler_file, 'rb') as f:
                        self.scalers[model_name] = pickle.load(f)
            
            # Load performance metrics
            performance_file = self.model_save_dir / f"performance_{timestamp}.json"
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    performance_data = json.load(f)
                
                for model_name, perf_dict in performance_data.items():
                    self.model_performance[model_name] = ModelPerformance(**perf_dict)
            
            # Load metadata
            metadata_file = self.model_save_dir / f"metadata_{timestamp}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.best_model_name = metadata.get('best_model')
                if metadata.get('training_time'):
                    self.last_training_time = datetime.fromisoformat(metadata['training_time'])
            
            logger.info(f"Models loaded from timestamp: {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance."""
        summary = {
            'best_model': self.best_model_name,
            'total_training_samples': len(self.training_data),
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'is_training': self.is_training,
            'models_trained': len(self.models),
            'model_performance': {}
        }
        
        for model_name, performance in self.model_performance.items():
            summary['model_performance'][model_name] = {
                'accuracy': performance.accuracy,
                'auc_score': performance.auc_score,
                'f1_score': performance.f1_score,
                'cross_val_score': performance.cross_val_score,
                'training_samples': performance.training_samples
            }
        
        return summary
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model."""
        try:
            target_model = model_name or self.best_model_name
            if not target_model or target_model not in self.models:
                return None
            
            model = self.models[target_model]
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create feature names (simplified)
                feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                return dict(zip(feature_names, importances))
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None


# NOTE: Do NOT use a global instance. Create ModelTrainer
# explicitly and inject the FeatureEngineer dependency.