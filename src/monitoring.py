"""
Monitoring and Logging System for ML Model
Tracks performance metrics, data drift, and system health
"""

import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import stats
from collections import deque
import logging

from pathlib import Path
Path('logs').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self, window_size=100, drift_threshold=0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Metrics storage
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.feature_stats = {}
        
        # Reference statistics for drift detection
        self.reference_stats = None
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        Path('logs/metrics').mkdir(exist_ok=True)
        
        logger.info("ModelMonitor initialized")
    
    def log_prediction(self, features, prediction, confidence, response_time):
        """
        Log a prediction for monitoring
        
        Args:
            features: dict of feature values
            prediction: model prediction (0 or 1)
            confidence: prediction confidence
            response_time: inference time in seconds
        """
        timestamp = datetime.now().isoformat()
        
        # Store metrics
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.response_times.append(response_time)
        
        # Log to file
        log_entry = {
            'timestamp': timestamp,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'response_time_ms': float(response_time * 1000),
            'features': features
        }
        
        # Append to daily log file
        log_file = f"logs/metrics/predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Update feature statistics
        self._update_feature_stats(features)
        
        # Check for drift periodically
        if len(self.predictions) == self.window_size:
            self._check_drift()
    
    def _update_feature_stats(self, features):
        """Update running statistics for features"""
        for feature_name, value in features.items():
            if feature_name not in self.feature_stats:
                self.feature_stats[feature_name] = {
                    'values': deque(maxlen=self.window_size),
                    'mean': 0,
                    'std': 0
                }
            
            self.feature_stats[feature_name]['values'].append(value)
            values = list(self.feature_stats[feature_name]['values'])
            self.feature_stats[feature_name]['mean'] = np.mean(values)
            self.feature_stats[feature_name]['std'] = np.std(values)
    
    def set_reference_statistics(self, reference_features):
        """
        Set reference statistics for drift detection
        
        Args:
            reference_features: dict of {feature_name: list of values}
        """
        self.reference_stats = {}
        for feature_name, values in reference_features.items():
            self.reference_stats[feature_name] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        logger.info("Reference statistics set for drift detection")
    
    def _check_drift(self):
        """Check for data drift using statistical tests"""
        if not self.reference_stats:
            return
        
        drift_detected = False
        drift_features = []
        
        for feature_name, stats in self.feature_stats.items():
            if feature_name not in self.reference_stats:
                continue
            
            current_values = list(stats['values'])
            ref_mean = self.reference_stats[feature_name]['mean']
            ref_std = self.reference_stats[feature_name]['std']
            
            # Use Kolmogorov-Smirnov test for distribution drift
            # Compare current distribution to reference
            current_mean = np.mean(current_values)
            current_std = np.std(current_values)
            
            # Simple drift detection: check if mean has shifted significantly
            z_score = abs(current_mean - ref_mean) / (ref_std + 1e-8)
            
            if z_score > 2:  # 2 standard deviations
                drift_detected = True
                drift_features.append({
                    'feature': feature_name,
                    'z_score': float(z_score),
                    'ref_mean': float(ref_mean),
                    'current_mean': float(current_mean)
                })
        
        if drift_detected:
            logger.warning(f"Data drift detected in features: {[f['feature'] for f in drift_features]}")
            self._log_drift_event(drift_features)
    
    def _log_drift_event(self, drift_features):
        """Log drift detection event"""
        drift_event = {
            'timestamp': datetime.now().isoformat(),
            'type': 'data_drift',
            'features': drift_features
        }
        
        with open('logs/drift_events.jsonl', 'a') as f:
            f.write(json.dumps(drift_event) + '\n')
    
    def get_metrics(self):
        """Get current performance metrics"""
        if not self.predictions:
            return None
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'prediction_distribution': {
                'up': sum(self.predictions) / len(self.predictions),
                'down': 1 - (sum(self.predictions) / len(self.predictions))
            },
            'confidence': {
                'mean': float(np.mean(self.confidences)),
                'std': float(np.std(self.confidences)),
                'min': float(np.min(self.confidences)),
                'max': float(np.max(self.confidences))
            },
            'response_time_ms': {
                'mean': float(np.mean(self.response_times) * 1000),
                'p50': float(np.percentile(self.response_times, 50) * 1000),
                'p95': float(np.percentile(self.response_times, 95) * 1000),
                'p99': float(np.percentile(self.response_times, 99) * 1000)
            },
            'total_predictions': len(self.predictions)
        }
        
        return metrics
    
    def export_metrics(self, filepath='logs/current_metrics.json'):
        """Export current metrics to file"""
        metrics = self.get_metrics()
        if metrics:
            with open(filepath, 'w') as f:
                json.dump(metrics, indent=2, fp=f)
            logger.info(f"Metrics exported to {filepath}")
        return metrics


class HealthChecker:
    """Check system health and model status"""
    
    def __init__(self):
        self.start_time = time.time()
        self.health_checks = []
        logger.info("HealthChecker initialized")
    
    def check_model_loaded(self, model):
        """Check if model is loaded correctly"""
        return model is not None
    
    def check_disk_space(self, min_gb=1):
        """Check available disk space"""
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        return free_gb >= min_gb
    
    def check_memory_usage(self, max_percent=90):
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < max_percent
        except ImportError:
            return True  # Skip if psutil not available
    
    def get_uptime(self):
        """Get service uptime in seconds"""
        return time.time() - self.start_time
    
    def get_health_status(self, model=None):
        """Get comprehensive health status"""
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': self.get_uptime(),
            'checks': {
                'model_loaded': self.check_model_loaded(model),
                'disk_space': self.check_disk_space(),
                'memory_usage': self.check_memory_usage()
            }
        }
        
        # Overall health
        if not all(status['checks'].values()):
            status['status'] = 'unhealthy'
            logger.warning("Health check failed")
        
        return status


class MetricsAggregator:
    """Aggregate and analyze metrics over time"""
    
    @staticmethod
    def load_daily_metrics(date_str):
        """Load metrics for a specific date"""
        log_file = f"logs/metrics/predictions_{date_str}.jsonl"
        
        if not Path(log_file).exists():
            return None
        
        predictions = []
        with open(log_file, 'r') as f:
            for line in f:
                predictions.append(json.loads(line))
        
        return predictions
    
    @staticmethod
    def analyze_daily_performance(date_str):
        """Analyze model performance for a day"""
        predictions = MetricsAggregator.load_daily_metrics(date_str)
        
        if not predictions:
            return None
        
        confidences = [p['confidence'] for p in predictions]
        response_times = [p['response_time_ms'] for p in predictions]
        pred_values = [p['prediction'] for p in predictions]
        
        analysis = {
            'date': date_str,
            'total_predictions': len(predictions),
            'predictions': {
                'up': sum(pred_values),
                'down': len(pred_values) - sum(pred_values),
                'up_percentage': sum(pred_values) / len(pred_values) * 100
            },
            'confidence': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences)
            },
            'performance': {
                'mean_response_ms': np.mean(response_times),
                'p95_response_ms': np.percentile(response_times, 95),
                'p99_response_ms': np.percentile(response_times, 99)
            }
        }
        
        return analysis
    
    @staticmethod
    def generate_report(days=7):
        """Generate performance report for last N days"""
        from datetime import timedelta
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_days': days,
            'daily_stats': []
        }
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
            stats = MetricsAggregator.analyze_daily_performance(date)
            if stats:
                report['daily_stats'].append(stats)
        
        # Save report
        report_file = f"logs/performance_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report generated: {report_file}")
        return report


# Example usage
if __name__ == '__main__':
    # Initialize monitor
    monitor = ModelMonitor()
    
    # Simulate some predictions
    for i in range(150):
        features = {
            'price_change': np.random.randn() * 0.02,
            'price_ma_7': 45000 + np.random.randn() * 1000,
            'volume_ma_7': 25e9 + np.random.randn() * 1e9
        }
        
        prediction = np.random.randint(0, 2)
        confidence = 0.5 + np.random.rand() * 0.3
        response_time = 0.05 + np.random.rand() * 0.02
        
        monitor.log_prediction(features, prediction, confidence, response_time)
    
    # Get metrics
    metrics = monitor.get_metrics()
    print("\nCurrent Metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Export metrics
    monitor.export_metrics()
    
    print("\n✅ Monitoring system demo complete!")
