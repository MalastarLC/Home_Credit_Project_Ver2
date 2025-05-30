# --- START OF FILE API_Script.py ---

import pandas as pd
import numpy as np
import mlflow
import re
import logging # For better logging

from preprocessing_pipeline import prepare_input_data, agg_numeric, count_categorical, sanitize_lgbm_colname, predicting_scores
from flask import Flask, request, jsonify


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO) # Outputs to console

@app.route('/predict', methods=['POST'])
def predict():

    # --- MODIFICATION: Runtime Diagnostic ---
    app.logger.info("--- RUNTIME DIAGNOSTIC ---")
    app.logger.info(f"Current sys.path: {sys.path}")
    
    # Try to find where preprocessing_pipeline was imported from
    try:
        import preprocessing_pipeline # Ensure it's imported to get __file__
        module_path = preprocessing_pipeline.__file__
        app.logger.info(f"preprocessing_pipeline module was imported from: {module_path}")
        if os.path.exists(module_path):
            app.logger.info(f"Content of {module_path} (first 300 chars):")
            with open(module_path, 'r') as f_content:
                app.logger.info(f_content.read(300))
        else:
            app.logger.error(f"Module path {module_path} does not exist!")
    except Exception as e_diag:
        app.logger.error(f"Error during runtime diagnostic for preprocessing_pipeline: {e_diag}")
    app.logger.info("--- END RUNTIME DIAGNOSTIC ---")
    # --- END MODIFICATION ---

    app.logger.info("Received prediction request.")
    if not request.is_json:
        app.logger.error("Request is not JSON.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    app.logger.info(f"Received data keys: {list(data.keys())}")

    required_dataframes = [
        'current_app', 'bureau', 'bureau_balance', 
        'previous_application', 'POS_CASH_balance', 
        'installments_payments', 'credit_card_balance'
    ]

    dfs = {}
    try:
        for df_name in required_dataframes:
            if df_name not in data:
                app.logger.error(f"Missing DataFrame in request: {df_name}")
                return jsonify({"error": f"Missing DataFrame: {df_name}"}), 400
            # The data for each key should be a list of records (dicts)
            dfs[df_name] = pd.DataFrame(data[df_name])
            app.logger.info(f"Successfully loaded DataFrame: {df_name} with shape {dfs[df_name].shape}")
            if dfs[df_name].empty and df_name == 'current_app': # current_app is critical
                 app.logger.error(f"Critical DataFrame '{df_name}' is empty.")
                 return jsonify({"error": f"Critical DataFrame '{df_name}' cannot be empty."}), 400

    except Exception as e:
        app.logger.error(f"Error converting JSON to DataFrames: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error processing input JSON: {str(e)}"}), 400

    try:
        app.logger.info("Starting data preparation...")
        full_data = prepare_input_data(
            dfs['current_app'], 
            dfs['bureau'], 
            dfs['bureau_balance'],
            dfs['previous_application'], 
            dfs['POS_CASH_balance'],
            dfs['installments_payments'], 
            dfs['credit_card_balance']
        )
        app.logger.info(f"Data preparation complete. Shape of full_data: {full_data.shape}")

        app.logger.info("Starting score prediction...")
        # If using a globally loaded model, pass it here:
        # client_scores, likely, not_likely = predicting_scores(full_data, model=get_model())
        # Otherwise, predicting_scores will load its own model:
        client_scores, likely, not_likely = predicting_scores(full_data)
        app.logger.info("Score prediction complete.")

        # Convert DataFrames to JSON serializable format (list of records)
        response = {
            "client_with_scores": client_scores.to_dict(orient='records'),
            "likely_to_repay": likely.to_dict(orient='records'),
            "not_likely_to_repay": not_likely.to_dict(orient='records')
        }
        app.logger.info("Prediction successful. Sending response.")
        return jsonify(response), 200

    except ValueError as ve: # Catch specific errors like missing columns
        app.logger.error(f"ValueError during processing: {str(ve)}", exc_info=True)
        return jsonify({"error": f"Configuration or data error: {str(ve)}"}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during prediction: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

#if __name__ == '__main__':
    # Make sure MLFLOW_TRACKING_URI is set in your environment if not using default ./mlruns
    # e.g., export MLFLOW_TRACKING_URI=http://localhost:5000
    # Or, set it in Python code (less ideal for prod):
    # mlflow.set_tracking_uri("file:./mlruns") # or your server URI

#    app.run(host='0.0.0.0', port=5001, debug=True) # debug=False for production

# --- END OF FILE API_Script.py ---