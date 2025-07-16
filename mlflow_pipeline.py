import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import matplotlib.pyplot as plt

from codecarbon import EmissionsTracker

class FeatureStore:
    """Feature storage class"""
    
    def __init__(self, base_path="feature_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.features_path = self.base_path / "features"
        self.features_path.mkdir(exist_ok=True)
        self.metadata_file = self.base_path / "metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"feature_versions": {}}
    
    def save_metadata(self):
        """Save metadata"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def save_features(self, features_df, version_name, description=""):
        """Save feature version"""
        version_path = self.features_path / f"{version_name}.pkl"
        features_df.to_pickle(version_path)
        
        self.metadata["feature_versions"][version_name] = {
            "created_at": datetime.now().isoformat(),
            "description": description,
            "shape": list(features_df.shape),
            "columns": list(features_df.columns),
            "file_path": str(version_path)
        }
        self.save_metadata()
        print(f"Feature version '{version_name}' saved")
    
    def load_features(self, version_name):
        """Load feature version"""
        version_path = self.features_path / f"{version_name}.pkl"
        if not version_path.exists():
            raise FileNotFoundError(f"Feature version '{version_name}' does not exist")
        
        features_df = pd.read_pickle(version_path)
        print(f"Feature version '{version_name}' loaded, shape: {features_df.shape}")
        return features_df
    
    def list_versions(self):
        """List all feature versions"""
        return list(self.metadata["feature_versions"].keys())

class MLPipeline:
    """Machine learning pipeline class"""
    def __init__(self, experiment_name="athletes_prediction"):
        self.experiment_name = experiment_name
        self.feature_store = FeatureStore()
        self.setup_mlflow()
        self.create_plot_directory()
    
    def create_plot_directory(self):
        """Create plot directory"""
        self.plot_dir = Path("plots")
        self.plot_dir.mkdir(exist_ok=True)
        
        self.carbon_dir = Path("carbon_tracking")
        self.carbon_dir.mkdir(exist_ok=True)
    
    def setup_mlflow(self):
        """Setup MLflow"""
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess data"""
        print("Loading data...")
        df = pd.read_csv(file_path)
        print(f"Original data shape: {df.shape}")
        
        df_clean = df.dropna(subset=['gender', 'age']).copy()
        print(f"Cleaned data shape: {df_clean.shape}")
        
        return df_clean
    
    def create_feature_version_1(self, df):
        """Create feature version 1: Basic features"""
        
        features = df.copy()
        
        numeric_features = ['age', 'height', 'weight', 'candj', 'snatch', 'deadlift', 'backsq']
        
        if 'gender' in features.columns:
            le_gender = LabelEncoder()
            features['gender_encoded'] = le_gender.fit_transform(features['gender'].fillna('Unknown'))
        
        features = features.dropna(subset=['deadlift'])
        
        for col in numeric_features:
            if col in features.columns:
                features[col] = features[col].fillna(features[col].median())
        
        final_features = ['age', 'height', 'weight', 'gender_encoded']
        for col in ['candj', 'snatch', 'backsq']:
            if col in features.columns:
                final_features.append(col)
        
        feature_df = features[final_features + ['deadlift']].copy()
        feature_df = feature_df.dropna()
        
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna()
        
        for col in final_features:
            if col in feature_df.columns and feature_df[col].dtype in ['float64', 'int64']:
                Q1 = feature_df[col].quantile(0.25)
                Q3 = feature_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                feature_df = feature_df[(feature_df[col] >= lower_bound) & (feature_df[col] <= upper_bound)]
        
        print(f"Feature version 1 final shape: {feature_df.shape}")
        print(f"Feature columns: {final_features}")
        
        return feature_df, final_features
    
    def create_feature_version_2(self, df):
        """Create feature version 2: Advanced features"""
        features = df.copy()
        
        numeric_features = ['age', 'height', 'weight', 'candj', 'snatch', 'deadlift', 'backsq']
        
        if 'gender' in features.columns:
            le_gender = LabelEncoder()
            features['gender_encoded'] = le_gender.fit_transform(features['gender'].fillna('Unknown'))
        
        features = features.dropna(subset=['deadlift'])
        
        for col in numeric_features:
            if col in features.columns:
                features[col] = features[col].fillna(features[col].median())
        
        if 'height' in features.columns and 'weight' in features.columns:
            features['bmi'] = features['weight'] / ((features['height'] / 100) ** 2)
        
        strength_cols = [col for col in ['candj', 'snatch', 'backsq'] if col in features.columns]
        if strength_cols:
            features['total_strength'] = features[strength_cols].sum(axis=1)
            features['avg_strength'] = features[strength_cols].mean(axis=1)
        
        features['age_group'] = pd.cut(features['age'], bins=[0, 25, 35, 45, 100], labels=[0, 1, 2, 3])
        features['age_group'] = features['age_group'].cat.codes 
        
        if 'backsq' in features.columns:
            features['strength_weight_ratio'] = features['backsq'] / features['weight']
        
        final_features = ['age', 'height', 'weight', 'gender_encoded', 'bmi', 'age_group']
        
        for col in ['candj', 'snatch', 'backsq', 'total_strength', 'avg_strength', 'strength_weight_ratio']:
            if col in features.columns:
                final_features.append(col)
        
        feature_df = features[final_features + ['deadlift']].copy()
        feature_df = feature_df.dropna()
        
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna()
        
        for col in final_features:
            if col in feature_df.columns and feature_df[col].dtype in ['float64', 'int64']:
                Q1 = feature_df[col].quantile(0.25)
                Q3 = feature_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                feature_df = feature_df[(feature_df[col] >= lower_bound) & (feature_df[col] <= upper_bound)]
        
        print(f"Feature version 2 final shape: {feature_df.shape}")
        print(f"Feature columns: {final_features}")
        
        return feature_df, final_features
    
    def train_model(self, X_train, X_test, y_train, y_test, hyperparams, run_name):
        """Train model"""
        with mlflow.start_run(run_name=run_name):
            tracker = EmissionsTracker(
                project_name=f"ml_experiment_{run_name}",
                output_dir=str(self.carbon_dir),
                log_level='error'
            )
            tracker.start()
            
            mlflow.log_params(hyperparams)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("features_count", X_train.shape[1])
            
            model = RandomForestRegressor(**hyperparams, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            metrics = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean_r2': cv_mean,
                'cv_std_r2': cv_std
            }
            
            mlflow.log_metrics(metrics)
            
            emissions = tracker.stop()
            mlflow.log_metric("carbon_emissions_kg", emissions)
            
            mlflow.sklearn.log_model(model, "model")
            
            self.plot_predictions(y_test, y_pred_test, run_name)
            
            self.plot_feature_importance(model, X_train.columns, run_name)
            
            self.plot_residuals(y_test, y_pred_test, run_name)
            
            print(f"Experiment {run_name} completed")
            print(f"Test R²: {test_r2:.4f}")
            print(f"Test MSE: {test_mse:.4f}")
            print(f"Carbon emissions: {emissions:.6f} kg CO²")
            
            return model, metrics, emissions
    
    def plot_predictions(self, y_true, y_pred, run_name):
        """Plot prediction comparison"""
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Prediction Comparison - {run_name}')
        plt.grid(True, alpha=0.3)
        
        plot_path = self.plot_dir / f"predictions_{run_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
        
        plot_data = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred,
            'Residuals': y_true - y_pred
        })
        print(f"\nPrediction comparison statistics - {run_name}:")
        print(plot_data.describe())
    
    def plot_feature_importance(self, model, feature_names, run_name):
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['Importance'])
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {run_name}')
        plt.grid(True, axis='x', alpha=0.3)
        
        plot_path = self.plot_dir / f"feature_importance_{run_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
        
        print(f"\nFeature importance data - {run_name}:")
        print(importance_df.sort_values('Importance', ascending=False))
    
    def plot_residuals(self, y_true, y_pred, run_name):
        """Plot residuals"""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Scatter Plot')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution Histogram')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plot_dir / f"residuals_{run_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
        
        residual_data = pd.DataFrame({
            'Predicted': y_pred,
            'Residuals': residuals,
            'Absolute_Residuals': np.abs(residuals)
        })
        print(f"\nResiduals statistics - {run_name}:")
        print(residual_data.describe())
    
    def run_experiments(self, data_file):
        """Run complete experiments"""
        df = self.load_and_preprocess_data(data_file)
        
        feature_v1, features_v1 = self.create_feature_version_1(df)
        feature_v2, features_v2 = self.create_feature_version_2(df)
        
        self.feature_store.save_features(feature_v1, "version_1", "Basic features version")
        self.feature_store.save_features(feature_v2, "version_2", "Advanced features version")
        
        hyperparams_1 = {
            'n_estimators': 50,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        }
        
        hyperparams_2 = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
        
        experiments = [
            ("v1_hp1", feature_v1, features_v1, hyperparams_1),
            ("v1_hp2", feature_v1, features_v1, hyperparams_2),
            ("v2_hp1", feature_v2, features_v2, hyperparams_1),
            ("v2_hp2", feature_v2, features_v2, hyperparams_2)
        ]
        
        results = {}
        
        for exp_name, feature_df, feature_cols, hyperparams in experiments:
            print(f"\nRunning experiment: {exp_name}")
            
            X = feature_df[feature_cols]
            y = feature_df['deadlift']
            
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            model, metrics, emissions = self.train_model(
                X_train, X_test, y_train, y_test, hyperparams, exp_name
            )
            
            results[exp_name] = {
                'model': model,
                'metrics': metrics,
                'emissions': emissions,
                'feature_version': exp_name.split('_')[0],
                'hyperparams': hyperparams
            }
        
        self.compare_experiments(results)
        
        return results
    
    def compare_experiments(self, results):
        """Compare experiment results"""
        print("\n" + "=" * 50)
        print("Experiment Results Comparison")
        
        
        comparison_data = []
        for exp_name, result in results.items():
            comparison_data.append({
                'Experiment': exp_name,
                'Feature_Version': result['feature_version'],
                'Test_R2': result['metrics']['test_r2'],
                'Test_MSE': result['metrics']['test_mse'],
                'Test_MAE': result['metrics']['test_mae'],
                'CV_R2': result['metrics']['cv_mean_r2'],
                'Carbon_Emissions_kg': result['emissions']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nExperiment results comparison table:")
        print(comparison_df.round(6))
        
        self.plot_experiment_comparison(comparison_df)
        
        return comparison_df
    
    def plot_experiment_comparison(self, comparison_df):
        """Plot experiment comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Experiment Results Comparison', fontsize=16)
        
        axes[0, 0].bar(comparison_df['Experiment'], comparison_df['Test_R2'])
        axes[0, 0].set_title('Test R² Comparison')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].bar(comparison_df['Experiment'], comparison_df['Test_MSE'])
        axes[0, 1].set_title('Test MSE Comparison')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].bar(comparison_df['Experiment'], comparison_df['CV_R2'])
        axes[1, 0].set_title('Cross-Validation R² Comparison')
        axes[1, 0].set_ylabel('CV R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(comparison_df['Experiment'], comparison_df['Carbon_Emissions_kg'])
        axes[1, 1].set_title('Carbon Emissions Comparison')
        axes[1, 1].set_ylabel('Carbon Emissions (kg CO²)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plot_dir / "experiment_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
        
        print("\nDetailed experiment comparison data:")
        for col in ['Test_R2', 'Test_MSE', 'CV_R2', 'Carbon_Emissions_kg']:
            print(f"\n{col}:")
            print(comparison_df[['Experiment', col]].sort_values(col))

def main():
    """Main function"""
    pipeline = MLPipeline("athletes_deadlift_prediction")
    
    results = pipeline.run_experiments("athletes.csv")
    
    print(pipeline.feature_store.list_versions())

if __name__ == "__main__":
    main()