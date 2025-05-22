# taxi_fare_predictor.py
import pandas as pd
import pickle
import duckdb
import sys
import os

def predict_fares(model_path, input_file_path, db_path='taxi_predictions.db'):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    if not os.path.exists(input_file_path):
        print(f"Error: Input file {input_file_path} not found")
        return
    
    try:
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as file:
            model, features = pickle.load(file)
        
        print(f"Loading data from {input_file_path}...")
        data = pd.read_csv(input_file_path)
        
        if 'id' not in data.columns:
            print("Error: Input data must contain an 'id' column")
            return
        
        missing_features = [feat for feat in features if feat not in data.columns]
        if missing_features:
            print(f"Error: Missing features in input data: {missing_features}")
            return
        
        X = data[features]
        
        print("Making predictions...")
        predictions = model.predict(X)
        
        results = pd.DataFrame({
            'id': data['id'],
            'predicted_fare_category': predictions
        })
        
        category_labels = {
            0: 'Low (≤$10)',
            1: 'Medium ($10-$20)',
            2: 'High ($20-$30)',
            3: 'Very High (>$30)'
        }
        results['predicted_fare_description'] = results['predicted_fare_category'].map(category_labels)
        
        print(f"Storing results in database {db_path}...")
        conn = duckdb.connect(db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fare_predictions (
                id INTEGER,
                predicted_value INTEGER,
                prediction_description VARCHAR
            )
        """)
        
        conn.execute("DELETE FROM fare_predictions")
        
        conn.execute("""
            INSERT INTO fare_predictions
            SELECT 
                id, 
                predicted_fare_category, 
                predicted_fare_description
            FROM results
        """)
        
        print("Preview of predictions stored in database:")
        result = conn.execute("SELECT * FROM fare_predictions LIMIT 5").fetchall()
        for row in result:
            print(row)
        
        conn.close()
        print(f"Predictions for {len(results)} samples successfully stored in {db_path}")
        
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python taxi_fare_predictor.py <model_file> <input_file> [db_file]")
        print("  model_file: Path to the pickled model file")
        print("  input_file: Path to the CSV file with samples to classify")
        print("  db_file: (Optional) Path to the DuckDB database file (default: taxi_predictions.db)")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_file_path = sys.argv[2]
    db_path = sys.argv[3] if len(sys.argv) > 3 else 'taxi_predictions.db'
    
    predict_fares(model_path, input_file_path, db_path)
