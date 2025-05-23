# --- API Script ---

import pandas as pd
import numpy as np
import mlflow
import re

from preprocessing_pipeline import prepare_input_data, agg_numeric, count_categorical, sanitize_lgbm_colname, predicting_scores
from flask import Flask, request, jsonify


MODEL_NAME = "HomeCredit_Default_Risk_Scoring_Model_v3"
MODEL_STAGE = "Production" # Or "Staging"
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
try:
    loaded_model_pipeline = mlflow.sklearn.load_model(model_uri)
    print(f"Successfully loaded model '{MODEL_NAME}' version from stage '{MODEL_STAGE}'")
except Exception as e:
    print(f"Error loading model from MLflow Registry: {e}")
    # Handle error: API might not be able to serve predictions
    loaded_model_pipeline = None


app = Flask(__name__)


