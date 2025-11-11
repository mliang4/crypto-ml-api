"""
Model Registry and Versioning System
Tracks model versions, metadata, and deployment history

Save this as: src/model_registry.py
"""

import json
import joblib
import shutil
from datetime import datetime
from pathlib import Path
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized model registry for version control and deployment tracking
    """
    
    def __init__(self, registry_path='models/registry'):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / 'registry.json'
        self.registry = self._load_registry()
        logger.info(f"Model registry initialized at {registry_path}")
    
    def _load_registry(self):
        """Load existing registry or create new one"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {
            'models': [],
            'production_model': None,
            'staging_model': None
        }
    
    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _calculate_checksum(self, filepath):
        """Calculate MD5 checksum of file"""
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
    
    def register_model(self, model_path, metadata, tags=None):
        """
        Register a new model version
        
        Args:
            model_path: path to model file
            metadata: dict with model info (accuracy, features, etc.)
            tags: list of tags (e.g., ['experiment', 'baseline'])
        
        Returns:
            version_id: unique version identifier
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version ID
        version_id = f"v{len(self.registry['models']) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create version directory
        version_dir = self.registry_path / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Copy model file
        model_filename = model_path.name
        new_model_path = version_dir / model_filename
        shutil.copy(model_path, new_model_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(new_model_path)
        
        # Create version entry
        version_entry = {
            'version_id': version_id,
            'timestamp': datetime.now().isoformat(),
            'model_path': str(new_model_path),
            'model_filename': model_filename,
            'checksum': checksum,
            'metadata': metadata,
            'tags': tags or [],
            'status': 'registered',
            'deployment_history': []
        }
        
        # Add to registry
        self.registry['models'].append(version_entry)
        self._save_registry()
        
        logger.info(f"Model registered: {version_id}")
        return version_id
    
    def promote_to_staging(self, version_id):
        """Promote model to staging environment"""
        model = self._get_model(version_id)
        if not model:
            raise ValueError(f"Model version not found: {version_id}")
        
        self.registry['staging_model'] = version_id
        model['status'] = 'staging'
        model['deployment_history'].append({
            'environment': 'staging',
            'timestamp': datetime.now().isoformat(),
            'action': 'promoted'
        })
        
        self._save_registry()
        logger.info(f"Model {version_id} promoted to staging")
    
    def promote_to_production(self, version_id):
        """Promote model to production environment"""
        model = self._get_model(version_id)
        if not model:
            raise ValueError(f"Model version not found: {version_id}")
        
        # Archive previous production model
        if self.registry['production_model']:
            prev_model = self._get_model(self.registry['production_model'])
            if prev_model:
                prev_model['status'] = 'archived'
        
        self.registry['production_model'] = version_id
        model['status'] = 'production'
        model['deployment_history'].append({
            'environment': 'production',
            'timestamp': datetime.now().isoformat(),
            'action': 'promoted'
        })
        
        self._save_registry()
        logger.info(f"Model {version_id} promoted to production")
    
    def rollback_production(self):
        """Rollback to previous production model"""
        production_models = [
            m for m in self.registry['models']
            if any(d['environment'] == 'production' for d in m['deployment_history'])
        ]
        
        if len(production_models) < 2:
            raise ValueError("No previous production model to rollback to")
        
        # Get second-to-last production model
        production_models.sort(
            key=lambda x: [d['timestamp'] for d in x['deployment_history'] 
                          if d['environment'] == 'production'][-1],
            reverse=True
        )
        
        previous_model = production_models[1]
        self.promote_to_production(previous_model['version_id'])
        logger.info(f"Rolled back to {previous_model['version_id']}")
    
    def _get_model(self, version_id):
        """Get model entry by version ID"""
        for model in self.registry['models']:
            if model['version_id'] == version_id:
                return model
        return None
    
    def get_production_model(self):
        """Get current production model info"""
        if not self.registry['production_model']:
            return None
        return self._get_model(self.registry['production_model'])
    
    def get_staging_model(self):
        """Get current staging model info"""
        if not self.registry['staging_model']:
            return None
        return self._get_model(self.registry['staging_model'])
    
    def list_models(self, status=None, tags=None):
        """
        List models with optional filtering
        
        Args:
            status: filter by status ('registered', 'staging', 'production', 'archived')
            tags: filter by tags
        """
        models = self.registry['models']
        
        if status:
            models = [m for m in models if m['status'] == status]
        
        if tags:
            models = [m for m in models if any(tag in m['tags'] for tag in tags)]
        
        return models
    
    def compare_models(self, version_id1, version_id2):
        """Compare two model versions"""
        model1 = self._get_model(version_id1)
        model2 = self._get_model(version_id2)
        
        if not model1 or not model2:
            raise ValueError("One or both model versions not found")
        
        comparison = {
            'version_1': {
                'version_id': model1['version_id'],
                'timestamp': model1['timestamp'],
                'metadata': model1['metadata'],
                'status': model1['status']
            },
            'version_2': {
                'version_id': model2['version_id'],
                'timestamp': model2['timestamp'],
                'metadata': model2['metadata'],
                'status': model2['status']
            }
        }
        
        # Compare metrics if available
        if 'accuracy' in model1['metadata'] and 'accuracy' in model2['metadata']:
            comparison['performance_diff'] = {
                'accuracy': model2['metadata']['accuracy'] - model1['metadata']['accuracy']
            }
        
        return comparison
    
    def delete_model(self, version_id):
        """Delete a model version (not recommended for production)"""
        model = self._get_model(version_id)
        if not model:
            raise ValueError(f"Model version not found: {version_id}")
        
        if model['status'] in ['production', 'staging']:
            raise ValueError(f"Cannot delete {model['status']} model. Demote first.")
        
        # Remove files
        version_dir = Path(model['model_path']).parent
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # Remove from registry
        self.registry['models'] = [m for m in self.registry['models'] 
                                   if m['version_id'] != version_id]
        self._save_registry()
        
        logger.info(f"Model deleted: {version_id}")
    
    def export_registry_report(self, output_path='models/registry_report.json'):
        """Export comprehensive registry report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_models': len(self.registry['models']),
            'production_model': self.get_production_model(),
            'staging_model': self.get_staging_model(),
            'models_by_status': {
                'registered': len(self.list_models(status='registered')),
                'staging': len(self.list_models(status='staging')),
                'production': len(self.list_models(status='production')),
                'archived': len(self.list_models(status='archived'))
            },
            'all_models': self.registry['models']
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Registry report exported to {output_path}")
        return report


# Example usage and testing
if __name__ == '__main__':
    print("="*80)
    print("Model Registry Demo")
    print("="*80)
    
    # Initialize registry
    registry = ModelRegistry()
    
    # Simulate registering models
    print("\n1. Registering models...")
    
    # Create dummy model files for demo
    Path('models').mkdir(exist_ok=True)
    dummy_model_1 = Path('models/temp_model_1.joblib')
    dummy_model_2 = Path('models/temp_model_2.joblib')
    dummy_model_1.touch()
    dummy_model_2.touch()
    
    # Register Model 1
    metadata1 = {
        'accuracy': 0.65,
        'model_type': 'RandomForest',
        'features': ['price_change', 'volume', 'rsi'],
        'training_date': '2024-11-01'
    }
    version1 = registry.register_model(dummy_model_1, metadata1, tags=['baseline'])
    print(f"   ✓ Registered {version1}")
    
    # Register Model 2
    metadata2 = {
        'accuracy': 0.68,
        'model_type': 'XGBoost',
        'features': ['price_change', 'volume', 'rsi', 'macd'],
        'training_date': '2024-11-11'
    }
    version2 = registry.register_model(dummy_model_2, metadata2, tags=['improved'])
    print(f"   ✓ Registered {version2}")
    
    # Promote to staging
    print(f"\n2. Promoting {version2} to staging...")
    registry.promote_to_staging(version2)
    
    # Promote to production
    print(f"\n3. Promoting {version2} to production...")
    registry.promote_to_production(version2)
    
    # Get production model
    print("\n4. Current production model:")
    prod_model = registry.get_production_model()
    if prod_model:
        print(f"   Version: {prod_model['version_id']}")
        print(f"   Accuracy: {prod_model['metadata']['accuracy']}")
        print(f"   Type: {prod_model['metadata']['model_type']}")
    
    # List all models
    print(f"\n5. All registered models:")
    for model in registry.list_models():
        status_icon = "🟢" if model['status'] == 'production' else "🟡" if model['status'] == 'staging' else "⚪"
        print(f"   {status_icon} {model['version_id']}: {model['status']} - Acc: {model['metadata']['accuracy']}")
    
    # Compare models
    print(f"\n6. Comparing {version1} vs {version2}:")
    comparison = registry.compare_models(version1, version2)
    if 'performance_diff' in comparison:
        acc_diff = comparison['performance_diff']['accuracy']
        print(f"   Accuracy improvement: {acc_diff:+.3f} ({acc_diff*100:+.1f}%)")
    
    # Export report
    print("\n7. Exporting registry report...")
    registry.export_registry_report()
    
    # Cleanup dummy files
    dummy_model_1.unlink(missing_ok=True)
    dummy_model_2.unlink(missing_ok=True)
    
    print("\n" + "="*80)
    print("✅ Model Registry Demo Complete!")
    print("="*80)
    print("\nRegistry file created at: models/registry/registry.json")
    print("Check it out to see the full version history!")
