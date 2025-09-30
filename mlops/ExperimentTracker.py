import mlflow
import mlflow.pytorch
from pathlib import Path
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name: str = "/Users/colizu2020@gmail.com/cifar100"):
        # Set Databricks as tracking URI
        mlflow.set_tracking_uri("databricks")
        
        # Set or create experiment
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def start_run(self, run_name: str = None):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"cifar100_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: dict):
        """Log hyperparameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log training metrics"""
        for key, value in metrics.items():
            if step is not None:
                mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metric(key, value)
    
    def log_model(self, model, artifact_path: str = "model", registered_model_name: str = None):
        """
        Log PyTorch model with proper parameter separation
        
        Args:
            model: The trained PyTorch model
            artifact_path: Where to store model files in the run (simple path)
            registered_model_name: Name for Unity Catalog registration (catalog.schema.model_name)
        """
        try:
            if registered_model_name:
                # Register model in Unity Catalog
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,  # Simple path like "model"
                    registered_model_name=registered_model_name  # Unity Catalog name
                )
                print(f"✅ Model logged and registered as: {registered_model_name}")
            else:
                # Just log as artifact without registration
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path
                )
                print(f"✅ Model logged as artifact at: {artifact_path}")
                
        except Exception as e:
            print(f"❌ Model logging failed: {e}")
            # Fallback: basic artifact logging
            try:
                mlflow.pytorch.log_model(pytorch_model=model, artifact_path=artifact_path)
                print(f"✅ Fallback: Model logged as artifact only")
            except Exception as fallback_error:
                print(f"❌ All model logging failed: {fallback_error}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log files/artifacts"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_figure(self, figure, filename: str):
        """Log matplotlib figures"""
        import tempfile
        import matplotlib.pyplot as plt
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            figure.savefig(tmp.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(tmp.name, f"plots/{filename}")
        
        plt.close(figure)  # Clean up