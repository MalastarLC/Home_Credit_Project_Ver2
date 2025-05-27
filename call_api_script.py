# --- START OF FILE test_api.py ---

import requests
import pandas as pd
import numpy as np
import json
import os

# API endpoint URL
#API_URL = "http://localhost:5001/predict" # Make sure port matches api.py
API_URL = "https://maxime-scorer-api-v3-b70c48541b06.herokuapp.com/" #New 

output_dir = "api_results"
os.makedirs(output_dir, exist_ok=True)

# --- Load Sample Data ---
# Adjust paths and filenames as necessary
# Ensure these files exist and are representative.
# For a single prediction, current_app_sample.csv might have one row.
# Other files should contain data related to the SK_ID_CURR(s) in current_app_sample.csv
# or be empty if that's a valid scenario your preprocessing handles.

# --- Load Sample Data ---
data_path = "data/"
try:
    current_app_df = pd.read_csv(f"{data_path}application_test.csv")
    bureau_df = pd.read_csv(f"{data_path}bureau.csv")
    bureau_balance_df = pd.read_csv(f"{data_path}bureau_balance.csv")
    previous_application_df = pd.read_csv(f"{data_path}previous_application.csv")
    pos_cash_balance_df = pd.read_csv(f"{data_path}POS_CASH_balance.csv")
    installments_payments_df = pd.read_csv(f"{data_path}installments_payments.csv")
    credit_card_balance_df = pd.read_csv(f"{data_path}credit_card_balance.csv")
    print("Full data loaded successfully.")

    frac_sample = 0.01
    print(f"Sampling data with fraction: {frac_sample}")
    current_app_df = current_app_df.sample(frac=frac_sample, random_state=42)
    
    sampled_sk_id_curr = current_app_df['SK_ID_CURR'].unique()
    
    bureau_df = bureau_df[bureau_df['SK_ID_CURR'].isin(sampled_sk_id_curr)]
    sampled_sk_id_bureau = bureau_df['SK_ID_BUREAU'].unique()
    bureau_balance_df = bureau_balance_df[bureau_balance_df['SK_ID_BUREAU'].isin(sampled_sk_id_bureau)]
    previous_application_df = previous_application_df[previous_application_df['SK_ID_CURR'].isin(sampled_sk_id_curr)]
    
    sampled_sk_id_prev_pa = previous_application_df['SK_ID_PREV'].unique()

    filter_pos = pos_cash_balance_df['SK_ID_CURR'].isin(sampled_sk_id_curr)
    if 'SK_ID_PREV' in pos_cash_balance_df.columns and len(sampled_sk_id_prev_pa) > 0 : # check if sampled_sk_id_prev_pa is not empty
        filter_pos |= pos_cash_balance_df['SK_ID_PREV'].isin(sampled_sk_id_prev_pa)
    pos_cash_balance_df = pos_cash_balance_df[filter_pos]

    filter_installments = installments_payments_df['SK_ID_CURR'].isin(sampled_sk_id_curr)
    if 'SK_ID_PREV' in installments_payments_df.columns and len(sampled_sk_id_prev_pa) > 0 :
        filter_installments |= installments_payments_df['SK_ID_PREV'].isin(sampled_sk_id_prev_pa)
    installments_payments_df = installments_payments_df[filter_installments]

    filter_credit_card = credit_card_balance_df['SK_ID_CURR'].isin(sampled_sk_id_curr)
    if 'SK_ID_PREV' in credit_card_balance_df.columns and len(sampled_sk_id_prev_pa) > 0 :
        filter_credit_card |= credit_card_balance_df['SK_ID_PREV'].isin(sampled_sk_id_prev_pa)
    credit_card_balance_df = credit_card_balance_df[filter_credit_card]

    print("Data sampled successfully.")
    print(f"current_app_df shape after sampling: {current_app_df.shape}")
    if current_app_df.empty:
        print("WARNING: current_app_df is empty after sampling.")

    all_dfs_dict = {
        "current_app": current_app_df,
        "bureau": bureau_df,
        "bureau_balance": bureau_balance_df,
        "previous_application": previous_application_df,
        "POS_CASH_balance": pos_cash_balance_df,
        "installments_payments": installments_payments_df,
        "credit_card_balance": credit_card_balance_df,
    }

    payload = {}
    print("\n--- Processing DataFrames for JSON serialization ---")
    for name, df_orig in all_dfs_dict.items():
        print(f"Processing DataFrame: {name} (shape: {df_orig.shape})")
        if df_orig.empty:
            print(f"DataFrame {name} is empty. Converting to empty list.")
            payload[name] = []
            continue

        df = df_orig.copy()

        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(str).replace(['inf', '-inf', 'Infinity', '-Infinity', 'NaN', 'nan', 'None', 'null', 'NA', '<NA>'], np.nan, regex=False)
                except Exception as e:
                    print(f"Warning: Could not process object column {col} in {name} with string replacements: {e}")

        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty: # Check if there are any numeric columns
             df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        for col in numeric_cols: # Iterate through original numeric_cols list
            # Ensure column still exists and is numeric before checking for inf
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if np.isinf(df[col]).any(): # Corrected: use np.isinf()
                    print(f"WARNING: DataFrame '{name}', Column '{col}' still contains Inf values AFTER replacement attempt!")
                    print(df[np.isinf(df[col])][col].head())

        df_none = df.astype(object).where(pd.notnull(df), None)
        
        try:
            payload[name] = df_none.to_dict(orient='records')
            print(f"Successfully converted {name} to dict.")
        except Exception as e_dict:
            print(f"ERROR converting DataFrame {name} to dict even after NaN/Inf handling: {e_dict}")
            raise

    print("\nPayload prepared.")

except FileNotFoundError as e:
    print(f"Error loading sample data: {e}")
    exit()
except pd.errors.EmptyDataError as e:
    print(f"Error: One of the sample CSV files is empty or malformed: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading/preparation: {str(e)}")
    traceback.print_exc() # Corrected: use traceback.print_exc()
    exit()

# --- Make API Request ---
print(f"\nSending request to {API_URL}...")
try:
    response = requests.post(API_URL, json=payload, timeout=300) 
    response.raise_for_status()
    print(f"Response Status Code: {response.status_code}")
    results = response.json()
    print("\n--- API Response ---")
    print(json.dumps(results, indent=4))
    if 'client_with_scores' in results:
        scores_df = pd.DataFrame(results['client_with_scores'])
        print("\n--- Scores DataFrame (from response) ---")
        print(scores_df.head())
except requests.exceptions.Timeout:
    print(f"\nRequest timed out.")
except requests.exceptions.ConnectionError as e:
    print(f"\nConnection Error: {e}")
except requests.exceptions.HTTPError as e:
    print(f"\nHTTP Error: {e}")
    print("Response content:", response.text)
except requests.exceptions.RequestException as e:
    print(f"\nAn error occurred during the request: {e}")
    print("Response text (if any):", response.text if hasattr(response, 'text') else "N/A")
except json.JSONDecodeError:
    print("\nError: Could not decode JSON response from API.")
    print("Response Text:", response.text)
except Exception as e:
    print(f"An unexpected error occurred during the API call: {str(e)}")
    traceback.print_exc() # Corrected: use traceback.print_exc()
    if 'response' in locals() and hasattr(response, 'text'):
        print("Response Text (if available):", response.text)

# --- SAVE RESULTS TO CSV ---
if 'client_with_scores' in results and results['client_with_scores']:
    client_scores_df = pd.DataFrame(results['client_with_scores'])
    output_path_all = os.path.join(output_dir, "client_scores_all.csv")
    client_scores_df.to_csv(output_path_all, index=False)
    print(f"\nAll client scores saved to: {output_path_all}")

if 'likely_to_repay' in results and results['likely_to_repay']:
    likely_df = pd.DataFrame(results['likely_to_repay'])
    output_path_likely = os.path.join(output_dir, "clients_likely_to_repay.csv")
    likely_df.to_csv(output_path_likely, index=False)
    print(f"Likely to repay clients saved to: {output_path_likely}")
    
if 'not_likely_to_repay' in results and results['not_likely_to_repay']:
    not_likely_df = pd.DataFrame(results['not_likely_to_repay'])
    output_path_not_likely = os.path.join(output_dir, "clients_not_likely_to_repay.csv")
    not_likely_df.to_csv(output_path_not_likely, index=False)
    print(f"Not likely to repay clients saved to: {output_path_not_likely}")




# --- END OF FILE test_api.py ---