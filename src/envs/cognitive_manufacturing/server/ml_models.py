"""Machine learning models service for predictive analytics."""

from __future__ import annotations
import numpy as np
from typing import Any, TYPE_CHECKING
from collections import deque
import json

if TYPE_CHECKING:
    from .database import DatabaseManager

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MLModelsService:
    """Manages machine learning models for predictive analytics."""

    def __init__(self):
        """Initialize ML models service."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for ML features. "
                "Install with: pip install scikit-learn numpy scipy"
            )

        # Models
        self.maintenance_model = None
        self.anomaly_detector = None
        self.quality_predictor = None
        self.scaler = StandardScaler()

        # RL Agent (simple Q-learning)
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.2

        # Historical data for forecasting
        self.demand_history = deque(maxlen=1000)

        # Training status
        self.models_trained = False

    # =====================================================================
    # Predictive Maintenance
    # =====================================================================

    def train_maintenance_model(self, sensor_data: list[dict]):
        """Train predictive maintenance model.

        Args:
            sensor_data: List of sensor readings with labels
                Each dict should have: temperature, vibration, wear_level, health_score, failed (bool)
        """
        if len(sensor_data) < 10:
            # Not enough data to train
            return False

        # Extract features and labels
        X = []
        y = []

        for reading in sensor_data:
            features = [
                reading.get('temperature', 0),
                reading.get('vibration', 0),
                reading.get('wear_level', 0),
                reading.get('health_score', 100),
                reading.get('speed', 0),
            ]
            X.append(features)
            y.append(1 if reading.get('failed', False) else 0)

        X = np.array(X)
        y = np.array(y)

        # Train scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train Random Forest
        self.maintenance_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        self.maintenance_model.fit(X_scaled, y)

        return True

    def predict_maintenance_need(
        self,
        temperature: float,
        vibration: float,
        wear_level: float,
        health_score: float,
        speed: float,
    ) -> dict:
        """Predict if maintenance is needed.

        Args:
            temperature: Current temperature
            vibration: Current vibration level
            wear_level: Current wear level
            health_score: Current health score
            speed: Current speed

        Returns:
            Dict with prediction results
        """
        if self.maintenance_model is None:
            # Model not trained, use heuristics
            maintenance_needed = (
                health_score < 60 or
                wear_level > 0.7 or
                temperature > 80
            )
            return {
                "maintenance_needed": maintenance_needed,
                "probability": 0.9 if maintenance_needed else 0.1,
                "hours_until_failure": 12.0 if maintenance_needed else 100.0,
                "confidence": 0.5,
                "method": "heuristic"
            }

        # Use trained model
        features = np.array([[temperature, vibration, wear_level, health_score, speed]])
        features_scaled = self.scaler.transform(features)

        probability = self.maintenance_model.predict_proba(features_scaled)[0][1]
        prediction = probability > 0.5

        # Estimate hours until failure based on wear rate and health
        hours_until_failure = max(1.0, (health_score / max(wear_level * 100, 1)) * 10)

        return {
            "maintenance_needed": bool(prediction),
            "probability": float(probability),
            "hours_until_failure": float(hours_until_failure),
            "confidence": float(max(probability, 1 - probability)),
            "method": "random_forest"
        }

    # =====================================================================
    # Anomaly Detection
    # =====================================================================

    def train_anomaly_detector(self, normal_data: list[dict]):
        """Train anomaly detection model.

        Args:
            normal_data: List of sensor readings during normal operation
        """
        if len(normal_data) < 20:
            return False

        # Extract features
        X = []
        for reading in normal_data:
            features = [
                reading.get('temperature', 0),
                reading.get('vibration', 0),
                reading.get('speed', 0),
                reading.get('production_output', 0),
            ]
            X.append(features)

        X = np.array(X)

        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # Expected proportion of outliers
            random_state=42
        )
        self.anomaly_detector.fit(X)

        return True

    def detect_anomalies(self, sensor_readings: list[dict]) -> dict:
        """Detect anomalies in sensor patterns.

        Args:
            sensor_readings: Recent sensor readings to analyze

        Returns:
            Dict with anomaly detection results
        """
        if len(sensor_readings) == 0:
            return {
                "anomaly_detected": False,
                "anomaly_score": 0.0,
                "anomalous_sensors": [],
                "severity": "none"
            }

        # Extract features
        X = []
        for reading in sensor_readings:
            features = [
                reading.get('temperature', 0),
                reading.get('vibration', 0),
                reading.get('speed', 0),
                reading.get('production_output', 0),
            ]
            X.append(features)

        X = np.array(X)

        if self.anomaly_detector is not None:
            # Use trained model
            predictions = self.anomaly_detector.predict(X)
            scores = self.anomaly_detector.score_samples(X)

            anomaly_count = np.sum(predictions == -1)
            anomaly_score = float(anomaly_count / len(predictions))

            # Identify which sensors are anomalous
            anomalous_sensors = []
            if anomaly_score > 0.3:
                # Check each sensor individually
                recent = sensor_readings[-1]
                historical_avg = {
                    'temperature': np.mean([r.get('temperature', 0) for r in sensor_readings[:-1]]),
                    'vibration': np.mean([r.get('vibration', 0) for r in sensor_readings[:-1]]),
                    'speed': np.mean([r.get('speed', 0) for r in sensor_readings[:-1]]),
                }

                for sensor, avg in historical_avg.items():
                    current = recent.get(sensor, 0)
                    if abs(current - avg) / max(avg, 1) > 0.3:  # 30% deviation
                        anomalous_sensors.append(sensor)

        else:
            # Fallback: use z-score method
            anomaly_score = 0.0
            anomalous_sensors = []

            for sensor in ['temperature', 'vibration', 'speed']:
                values = [r.get(sensor, 0) for r in sensor_readings]
                mean = np.mean(values)
                std = np.std(values)

                if std > 0:
                    z_score = abs((values[-1] - mean) / std)
                    if z_score > 2.5:  # 2.5 standard deviations
                        anomaly_score = max(anomaly_score, min(z_score / 3, 1.0))
                        anomalous_sensors.append(sensor)

        # Determine severity
        if anomaly_score > 0.7:
            severity = "high"
        elif anomaly_score > 0.4:
            severity = "medium"
        else:
            severity = "low"

        return {
            "anomaly_detected": anomaly_score > 0.3,
            "anomaly_score": float(anomaly_score),
            "anomalous_sensors": anomalous_sensors,
            "severity": severity
        }

    # =====================================================================
    # Quality Prediction
    # =====================================================================

    def train_quality_predictor(self, production_data: list[dict]):
        """Train quality prediction model.

        Args:
            production_data: List of production units with quality scores
                Each dict: speed, temperature, vibration, quality (0-1)
        """
        if len(production_data) < 10:
            return False

        # Extract features and target
        X = []
        y = []

        for unit in production_data:
            features = [
                unit.get('speed', 0),
                unit.get('temperature', 0),
                unit.get('vibration', 0),
                unit.get('wear_level', 0),
            ]
            X.append(features)
            y.append(unit.get('quality', 1.0))

        X = np.array(X)
        y = np.array(y)

        # Train Linear Regression
        self.quality_predictor = LinearRegression()
        self.quality_predictor.fit(X, y)

        return True

    def predict_quality(
        self,
        speed: float,
        temperature: float,
        vibration: float,
        wear_level: float,
    ) -> dict:
        """Predict product quality.

        Args:
            speed: Machine speed
            temperature: Operating temperature
            vibration: Vibration level
            wear_level: Machine wear level

        Returns:
            Dict with quality prediction
        """
        if self.quality_predictor is None:
            # Heuristic quality model
            quality = 1.0
            quality -= min(temperature / 100, 0.2)  # High temp reduces quality
            quality -= min(vibration / 0.5, 0.2)     # High vibration reduces quality
            quality -= min(wear_level, 0.2)         # Wear reduces quality
            quality = max(0, min(1, quality))

            return {
                "predicted_quality": float(quality),
                "confidence_interval": [quality - 0.1, quality + 0.1],
                "pass_probability": float(quality),
                "method": "heuristic"
            }

        # Use trained model
        features = np.array([[speed, temperature, vibration, wear_level]])
        predicted_quality = float(self.quality_predictor.predict(features)[0])
        predicted_quality = max(0, min(1, predicted_quality))

        # Simple confidence interval (would use prediction intervals in production)
        confidence_interval = [
            max(0, predicted_quality - 0.05),
            min(1, predicted_quality + 0.05)
        ]

        return {
            "predicted_quality": predicted_quality,
            "confidence_interval": confidence_interval,
            "pass_probability": predicted_quality,
            "method": "linear_regression"
        }

    # =====================================================================
    # RL Optimization
    # =====================================================================

    def get_state_key(self, state: dict) -> str:
        """Convert state dict to string key for Q-table."""
        # Discretize state for Q-learning
        health = int(state.get('health_score', 100) / 20) * 20  # Buckets: 0, 20, 40, 60, 80, 100
        temp = int(state.get('temperature', 20) / 10) * 10      # Buckets: 20, 30, 40, ...
        speed = int(state.get('speed', 0) / 25) * 25            # Buckets: 0, 25, 50, 75, 100

        return f"h{health}_t{temp}_s{speed}"

    def select_action_rl(
        self,
        state: dict,
        available_actions: list[dict],
        learning_mode: str = "exploit"
    ) -> dict:
        """Select action using RL policy.

        Args:
            state: Current environment state
            available_actions: List of possible actions
            learning_mode: "exploit" (use best action) or "explore" (try new actions)

        Returns:
            Selected action with expected reward
        """
        state_key = self.get_state_key(state)

        # Initialize Q-values for this state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = {
                json.dumps(action): 0.0 for action in available_actions
            }

        # Epsilon-greedy exploration
        if learning_mode == "explore" and np.random.random() < self.exploration_rate:
            # Explore: random action
            action = np.random.choice(available_actions)
            action_key = json.dumps(action)
            expected_reward = self.q_table[state_key].get(action_key, 0.0)
        else:
            # Exploit: best known action
            q_values = self.q_table[state_key]
            best_action_key = max(q_values, key=q_values.get)
            action = json.loads(best_action_key)
            expected_reward = q_values[best_action_key]

        return {
            "action": action,
            "expected_reward": float(expected_reward),
            "confidence": 0.8 if learning_mode == "exploit" else 0.3
        }

    def update_q_value(self, state: dict, action: dict, reward: float, next_state: dict):
        """Update Q-value after receiving reward (Q-learning update).

        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state after action
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        action_key = json.dumps(action)

        # Initialize if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0

        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state_key][action_key]

        # Max Q-value for next state
        if next_state_key in self.q_table and self.q_table[next_state_key]:
            max_next_q = max(self.q_table[next_state_key].values())
        else:
            max_next_q = 0.0

        # Update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state_key][action_key] = new_q

    # =====================================================================
    # Demand Forecasting
    # =====================================================================

    def update_demand_history(self, production_volume: int, timestamp: float):
        """Add data point to demand history."""
        self.demand_history.append({
            'timestamp': timestamp,
            'volume': production_volume
        })

    def forecast_demand(self, horizon: int = 24) -> list[dict]:
        """Forecast future demand.

        Args:
            horizon: Number of hours to forecast

        Returns:
            List of forecasts for each hour
        """
        if len(self.demand_history) < 10:
            # Not enough history, return simple projection
            avg_demand = 100  # Default
            if len(self.demand_history) > 0:
                avg_demand = np.mean([d['volume'] for d in self.demand_history])

            return [
                {
                    "hour": h,
                    "demand": float(avg_demand),
                    "lower_bound": float(avg_demand * 0.8),
                    "upper_bound": float(avg_demand * 1.2),
                }
                for h in range(1, horizon + 1)
            ]

        # Simple exponential smoothing
        values = [d['volume'] for d in self.demand_history]
        alpha = 0.3  # Smoothing factor

        # Calculate smoothed values
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])

        last_smoothed = smoothed[-1]

        # Detect trend
        recent_values = values[-min(20, len(values)):]
        trend = np.mean(np.diff(recent_values))

        # Generate forecast
        forecast = []
        for h in range(1, horizon + 1):
            point_forecast = last_smoothed + trend * h
            point_forecast = max(0, point_forecast)  # Non-negative

            # Simple confidence interval
            std = np.std(values[-min(50, len(values)):])
            lower = max(0, point_forecast - 1.96 * std)
            upper = point_forecast + 1.96 * std

            forecast.append({
                "hour": h,
                "demand": float(point_forecast),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            })

        return forecast
