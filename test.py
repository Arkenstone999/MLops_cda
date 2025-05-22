# Taxi Fare Prediction - Classification Approach with MLflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
import pickle
import warnings
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

warnings.filterwarnings('ignore')

# Set up MLflow experiment
mlflow.set_experiment("Taxi Fare Classification")

# 1. Load the dataset
df = pd.read_parquet('yellow_tripdata_2025-02.parquet')

# 2. Feature Engineering
# Create fare categories for classification
def categorize_fare(fare):
    if fare <= 10:
        return 0  # Low fare
    elif fare <= 20:
        return 1  # Medium fare
    elif fare <= 30:
        return 2  # High fare
    else:
        return 3  # Very high fare

df['fare_category'] = df['fare_amount'].apply(categorize_fare)

# Create time-based features
df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60

# Extract time features
df['hour_of_day'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['month'] = df['pickup_datetime'].dt.month

# Use trip_distance feature if it exists, otherwise calculate it
if 'trip_distance' in df.columns:
    distance_feature = 'trip_distance'
else:
    # Using Haversine formula to calculate distance
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lon1, lat1, lon2, lat2):
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r
    
    df['calculated_distance'] = df.apply(
        lambda x: haversine(
            x['pickup_longitude'], 
            x['pickup_latitude'], 
            x['dropoff_longitude'], 
            x['dropoff_latitude']
        ) if all(pd.notna([
            x['pickup_longitude'], 
            x['pickup_latitude'], 
            x['dropoff_longitude'], 
            x['dropoff_latitude']
        ])) else np.nan, 
        axis=1
    )
    
    distance_feature = 'calculated_distance'

# 3. Data Preprocessing
# Select features for model
features = [
    distance_feature,
    'passenger_count',
    'hour_of_day',
    'day_of_week',
    'month',
    'trip_duration'
]

# Handle potential missing values in selected features
for feature in features:
    if feature in df.columns and df[feature].isnull().sum() > 0:
        if df[feature].dtype in ['int64', 'float64']:
            df[feature] = df[feature].fillna(df[feature].median())
        else:
            df[feature] = df[feature].fillna(df[feature].mode()[0])

# Select rows with valid features and target
X = df[features].copy()
y = df['fare_category'].copy()

# Clean data - remove NaNs
valid_indices = ~(X.isnull().any(axis=1) | pd.isnull(y))
X = X[valid_indices]
y = y[valid_indices]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# 5. Build preprocessing pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features)
    ]
)

# 6. MLflow Model Training and Comparison
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
}

best_accuracy = 0
best_model_name = None

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"Basic-{model_name}"):
        # Create and train the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        if model_name == 'RandomForest':
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
        elif model_name == 'GradientBoosting':
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("learning_rate", model.learning_rate)
        elif model_name == 'LogisticRegression':
            mlflow.log_param("C", model.C)
            mlflow.log_param("solver", model.solver)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(pipeline, f"{model_name}_model", signature=signature)
        
        # Generate and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save and log figure
        cm_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = f"classification_report_{model_name}.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)
        
        # Print results
        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name

print(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")

# 7. Hyperparameter Tuning for the Best Model
with mlflow.start_run(run_name=f"Tuned-{best_model_name}"):
    if best_model_name == 'RandomForest':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        }
    elif best_model_name == 'GradientBoosting':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    else:  # LogisticRegression
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__penalty': ['l1', 'l2']
        }
    
    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', models[best_model_name])
    ])
    
    grid_search = GridSearchCV(
        best_pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Log training start
    mlflow.log_param("model_type", best_model_name)
    mlflow.log_param("param_grid", param_grid)
    
    # Perform grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and log them
    for param, value in grid_search.best_params_.items():
        param_name = param.split('__')[1]
        mlflow.log_param(f"best_{param_name}", value)
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Calculate and log metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    
    # Log the final tuned model
    signature = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(best_model, f"tuned_{best_model_name}", signature=signature)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Tuned {best_model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Save and log figure
    cm_path = f"confusion_matrix_tuned_{best_model_name}.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()
    
    # If feature importance is available, plot and log it
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        feature_importances = best_model.named_steps['classifier'].feature_importances_
        feature_names = X.columns
        
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Feature Importance - Tuned {best_model_name}')
        plt.tight_layout()
        
        # Save and log figure
        fi_path = f"feature_importance_tuned_{best_model_name}.png"
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path)
        plt.close()
    
    print(f"Tuned {best_model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print("Best parameters:", grid_search.best_params_)
    
    # Save the best model for later use
    model_filename = 'taxi_fare_classifier_mlflow.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump((best_model, features), file)
    
    mlflow.log_artifact(model_filename)
    print(f"Model has been exported to {model_filename}")
