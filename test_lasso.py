# Simple MLflow Example for Lasso (view UI quickly)
# ----------------------------------------------
# 1. Install MLflow if needed:
#    pip install mlflow
# 2. Run this script:
#    python simple_lasso_mlflow.py
# 3. In another terminal, start the UI:
#    mlflow ui --backend-store-uri mlruns
#    Open http://localhost:5000 in your browser

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split

# --- Data loading & preprocessing ---
df = pd.read_parquet('yellow_tripdata_2025-02.parquet')
df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
# Filter outliers and define target
df = df[(df.duration > 1) & (df.duration < 60)]
df['long_trip'] = (df.duration > 30).astype(int)

# Select minimal features that may have NaNs handled
features = ['trip_distance', 'passenger_count']
X = df[features]
y = df['long_trip']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- MLflow setup ---
mlflow.set_experiment('simple_lasso')

with mlflow.start_run():
    alphas = [0.01, 0.1, 1.0]
    mlflow.log_param('alphas', alphas)

    for idx, alpha in enumerate(alphas):
        # Build & evaluate with imputation
        pipe = make_pipeline(
            SimpleImputer(strategy='median'),  # handle NaNs
            StandardScaler(),
            Lasso(alpha=alpha, random_state=0)
        )
        score = cross_val_score(pipe, X_train, y_train, cv=3).mean()
        mlflow.log_metric('cv_score', score, step=idx)

    # Train final model using the best alpha
    cv_scores = [run.data.metrics['cv_score'] for run in mlflow.search_runs(filter_string="tags.mlflow.runName = 'simple_lasso'")['run_id']]
    best_alpha = alphas[int(np.argmax(cv_scores))]
    final_pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        Lasso(alpha=best_alpha, random_state=0)
    )
    final_pipe.fit(X_train, y_train)
    mlflow.sklearn.log_model(final_pipe, 'lasso_model')

print('Experiment complete.')
print('Start the MLflow UI with:')
print('  mlflow ui --backend-store-uri mlruns')
print('Then open http://localhost:5000 to explore runs.')
