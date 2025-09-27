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
    
    def log_model(self, model, model_name: str = "cifar100_model"):
        """Log PyTorch model"""
        mlflow.pytorch.log_model(
            model, 
            model_name,
            registered_model_name=f"{model_name}"
        )
    
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