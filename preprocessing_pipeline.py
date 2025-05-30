print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("EXECUTING THIS VERSION OF preprocessing_pipeline.py - V_DEBUG_001")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# --- Importation des librairies ---

import pandas as pd
import numpy as np

import mlflow
import re
import os # Ensure this is present
import traceback # MODIFICATION 1: ADD THIS IMPORT

from functools import reduce

from sklearn.preprocessing import LabelEncoder
# from memory_profiler import profile # Ensure this is commented/removed

# --- Définition des fonctions nécessaires à la préparation de l'input pour le modèle ---

# ... (agg_numeric function - NO CHANGES) ...
def agg_numeric(df, group_var, df_name):
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    columns = [group_var]
    for var in agg.columns.levels[0]:
        if var != group_var:
            for stat in agg.columns.levels[1][:-1]:
                columns.append('%s_%s_%s' % (df_name, var, stat))
    agg.columns = columns
    return agg

# ... (count_categorical function - NO CHANGES) ...
def count_categorical(df, group_var, df_name):
    categorical = pd.get_dummies(df.select_dtypes('object'))
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    column_names = []
    for var in categorical.columns.levels[0]:
        for stat in ['count', 'count_norm']:
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    categorical.columns = column_names
    return categorical

# ... (sanitize_lgbm_colname function - NO CHANGES) ...
def sanitize_lgbm_colname(colname):
    colname_str = str(colname)
    sanitized = re.sub(r'[\[\]{}":\',.<>\s/?!@#$%^&*()+=-]+', '_', colname_str)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    if not sanitized:
        sanitized = f"col_{hash(colname_str)}"
    if sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized

# --- Création de la fonction préparant les données d'entrée du modèle ---
# @profile # Ensure commented out
def prepare_input_data(current_app, bureau, bureau_balance, previous_application, POS_CASH_balance, installments_payments, credit_card_balance):
    """

    Receives application data and aggregates each DataFrame then combines them into the expected input for the scoring model

    Parameters
    --------

    current_app, bureau, bureau_balance, previous_application, POS_CASH_balance, installment_payments, credit_card_balance :
    The 7 DataFrames used for the feature engineering process

    Return
    --------
    client_data : dataframe
        A dataframe with the required features matching the model's expected input features
    
    """

    # --- Preparation de current_app ---

    # Label encoding des variables catégorielles avec 2 catégories uniques ou moins
    le = LabelEncoder()

    for col in current_app:
        if current_app[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(current_app[col].unique())) <= 2:
                # Train on the training data
                le.fit(current_app[col])
                # Transform both training and testing data
                current_app[col] = le.transform(current_app[col])
    
    # One hot encoding pour le reste des variables catégorielles
    current_app = pd.get_dummies(current_app, dtype='float64')

    # --- Aggrégation de bureau et bureau_balance ---
    
    # Création des features "manuelles"
    bureau_balance_loan_duration_months = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].count().reset_index()
    bureau_balance_loan_duration_months.rename(columns = {'MONTHS_BALANCE' : 'MONTHS_LOAN_DURATION'}, inplace=True)

    bureau_balance_last_known_loan_status = bureau_balance.sort_values(by='MONTHS_BALANCE', ascending=False).groupby('SK_ID_BUREAU').first().reset_index()[['SK_ID_BUREAU', 'STATUS']]
    #bureau_balance_last_known_loan_status = pd.get_dummies(bureau_balance_last_known_loan_status, dtype=int)
    all_expected_status_categories = ['0', '1', '2', '3', '4', '5', 'C', 'X'] # These are the raw STATUS values
    for cat_col_name in all_expected_status_categories:
        if cat_col_name not in bureau_balance_last_known_loan_status.columns:
            bureau_balance_last_known_loan_status[cat_col_name] = 0

    bureau_balance_for_dpd_flag = bureau_balance.copy()
    non_dpd_statuses = ['C', 'X', '0']
    dpd_flagged = ~bureau_balance_for_dpd_flag['STATUS'].isin(non_dpd_statuses) #Cree True/False value en fonction de la condition ici dpd --> True
    bureau_balance_for_dpd_flag['DPD_FLAG'] = dpd_flagged.astype(int)
    bureau_balance_nombre_delais_de_paiements = bureau_balance_for_dpd_flag.groupby(by='SK_ID_BUREAU')['DPD_FLAG'].sum().reset_index()

    bureau_balance_nombre_de_delais_de_paiements_par_categorie = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts().unstack(fill_value=0).reset_index()
    bureau_balance_nombre_de_delais_de_paiements_par_categorie.columns.name = None

    Values_to_map = {'C':0, 'X':0, '0':0, '1':15, '2':45, '3':75, '4':105, '5':120}
    bureau_balance_for_mean_delay_calc = bureau_balance.copy()
    bureau_balance_for_mean_delay_calc['MEAN_DAYS_PAST_DUE'] = bureau_balance_for_mean_delay_calc['STATUS'].map(Values_to_map).fillna('Unknown')
    bureau_balance_duree_moyenne_delais_de_paiements = bureau_balance_for_mean_delay_calc.groupby('SK_ID_BUREAU')['MEAN_DAYS_PAST_DUE'].mean().reset_index()

    bureau_balance_loan_duration_categorised = bureau_balance.sort_values(by='MONTHS_BALANCE', ascending=True).groupby('SK_ID_BUREAU').first().reset_index()
    bureau_balance_loan_duration_categorised['YEAR_LOAN_DURATION'] = round(bureau_balance_loan_duration_categorised['MONTHS_BALANCE']/(-12), 1)
    bureau_balance_loan_duration_categorised['LOAN_TYPE'] = np.where(
        bureau_balance_loan_duration_categorised['YEAR_LOAN_DURATION'] >= 5, # Condition
        'Long Term',                                             # Value if True
        'Short Term'                                             # Value if False
    )
    bureau_balance_loan_duration_categorised = bureau_balance_loan_duration_categorised.drop(columns=['MONTHS_BALANCE', 'STATUS'], axis=0)
    bureau_balance_loan_duration_categorised = pd.get_dummies(bureau_balance_loan_duration_categorised, dtype=int)

    dfs_to_merge = [bureau_balance_loan_duration_months,
    bureau_balance_last_known_loan_status,
    bureau_balance_nombre_delais_de_paiements,
    bureau_balance_nombre_de_delais_de_paiements_par_categorie,
    bureau_balance_duree_moyenne_delais_de_paiements,
    bureau_balance_loan_duration_categorised]

    merge_key = 'SK_ID_BUREAU'

    merge_function = lambda left_df, right_df: pd.merge(left_df, right_df, on=merge_key, how='inner')
    final_bureau_balance_features = reduce(merge_function, dfs_to_merge)
    #final_bureau_balance_features = final_bureau_balance_features.drop(columns=['0', 'C', 'X'])


    bureau_for_credit_status = bureau.copy()
    bureau_number_of_type_of_credits = bureau.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].value_counts().unstack(fill_value=0).reset_index()
    bureau_number_of_type_of_credits.columns.name = None

    bureau_for_encoding_credit_types = bureau[['SK_ID_CURR', 'CREDIT_TYPE']]
    bureau_credit_type_encoded = pd.get_dummies(bureau_for_encoding_credit_types, dtype=int)
    bureau_liste_colonnes_encodees = bureau_credit_type_encoded.columns
    bureau_liste_colonnes_a_agreger = [col for col in bureau_liste_colonnes_encodees if col != 'SK_ID_CURR']
    bureau_credit_type_encoded_and_aggregated = bureau_credit_type_encoded.groupby('SK_ID_CURR')[bureau_liste_colonnes_a_agreger].sum().reset_index()

    bureau_for_max_amount_overdue = bureau[['SK_ID_CURR', 'AMT_CREDIT_MAX_OVERDUE']]
    bureau_for_max_amount_overdue.fillna(value=0, inplace=True)
    bureau_for_max_amount_overdue = bureau_for_max_amount_overdue.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].max().reset_index()

    bureau_for_mean_amount_overdue_across_loans = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']]
    bureau_for_mean_amount_overdue_across_loans = bureau_for_mean_amount_overdue_across_loans.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].mean().reset_index()

    bureau_for_credit_prolonged_count = bureau[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']]
    bureau_for_credit_prolonged_count = bureau_for_credit_prolonged_count.groupby('SK_ID_CURR')['CNT_CREDIT_PROLONG'].sum().reset_index()

    bureau_for_proportions_repaid = bureau[['SK_ID_CURR', 'SK_ID_BUREAU', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM']]
    bureau_for_proportions_repaid.dropna(subset='AMT_CREDIT_SUM', inplace=True)
    bureau_for_proportions_repaid['AMT_CREDIT_SUM_DEBT_IS_MISSING'] = bureau_for_proportions_repaid['AMT_CREDIT_SUM_DEBT'].isnull().astype(int)
    mask_active_missing_debt = bureau_for_proportions_repaid['AMT_CREDIT_SUM_DEBT'].isnull()
    bureau_for_proportions_repaid.loc[mask_active_missing_debt, 'AMT_CREDIT_SUM_DEBT'] = bureau_for_proportions_repaid.loc[mask_active_missing_debt, 'AMT_CREDIT_SUM']
    bureau_for_proportions_repaid_with_nan_values_treated = bureau_for_proportions_repaid.copy()
    bureau_for_proportions_repaid_with_nan_values_treated = bureau_for_proportions_repaid_with_nan_values_treated.groupby('SK_ID_CURR')[['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM']].sum().reset_index()
    bureau_for_proportions_repaid_with_nan_values_treated['PROPORTION_REPAID'] = np.nan
    # Rule 1: SUM = 0 and DEBT = 0  => Repaid = 0
    mask_zero_both = (bureau_for_proportions_repaid_with_nan_values_treated['AMT_CREDIT_SUM'] == 0) & (bureau_for_proportions_repaid_with_nan_values_treated['AMT_CREDIT_SUM_DEBT'] == 0)
    bureau_for_proportions_repaid_with_nan_values_treated.loc[mask_zero_both, 'PROPORTION_REPAID'] = 1.0
    # Rule 2: DEBT < 0 => Repaid = 1
    mask_neg_debt = bureau_for_proportions_repaid_with_nan_values_treated['AMT_CREDIT_SUM_DEBT'] < 0
    bureau_for_proportions_repaid_with_nan_values_treated.loc[mask_neg_debt, 'PROPORTION_REPAID'] = 1.0
    # Rule 4: DEBT>0 and SUM = 0
    mask_pos_debt_null_sum = (bureau_for_proportions_repaid_with_nan_values_treated['AMT_CREDIT_SUM'] == 0) & (bureau_for_proportions_repaid_with_nan_values_treated['AMT_CREDIT_SUM_DEBT'] > 0)
    bureau_for_proportions_repaid_with_nan_values_treated.loc[mask_pos_debt_null_sum, 'PROPORTION_REPAID'] = 0
    # Rule 4: All other cases where SUM > 0
    mask_calculate = (bureau_for_proportions_repaid_with_nan_values_treated['AMT_CREDIT_SUM'] > 0) & (bureau_for_proportions_repaid_with_nan_values_treated['PROPORTION_REPAID'].isnull()) # Only calculate where not set yet
    bureau_for_proportions_repaid_with_nan_values_treated.loc[mask_calculate, 'PROPORTION_REPAID'] = 1.0 - (bureau_for_proportions_repaid_with_nan_values_treated.loc[mask_calculate, 'AMT_CREDIT_SUM_DEBT'] / bureau_for_proportions_repaid_with_nan_values_treated.loc[mask_calculate, 'AMT_CREDIT_SUM'])

    bureau_for_deadline_in_the_past = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE', 'DAYS_CREDIT_ENDDATE']].copy()
    bureau_for_deadline_in_the_past['LOAN_ACTIVE_PAST_DEADLINE'] = np.where(
        (bureau_for_deadline_in_the_past['CREDIT_ACTIVE'] == 'Active') & (bureau_for_deadline_in_the_past['DAYS_CREDIT_ENDDATE'] < 0), # Condition
        1,                                             # Value if True
        0                                             # Value if False
    )
    bureau_for_deadline_in_the_past = bureau_for_deadline_in_the_past.groupby('SK_ID_CURR')['LOAN_ACTIVE_PAST_DEADLINE'].sum().reset_index()

    bureau_for_mean_days_spent_overdue = bureau[['SK_ID_CURR', 'CREDIT_DAY_OVERDUE']]
    bureau_for_mean_days_spent_overdue = bureau_for_mean_days_spent_overdue.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].mean().reset_index()

    dfs_to_merge = [bureau_number_of_type_of_credits,
    bureau_credit_type_encoded_and_aggregated,
    bureau_for_max_amount_overdue,
    bureau_for_mean_amount_overdue_across_loans,
    bureau_for_credit_prolonged_count,
    bureau_for_proportions_repaid_with_nan_values_treated,
    bureau_for_deadline_in_the_past,
    bureau_for_mean_days_spent_overdue]

    merge_key = 'SK_ID_CURR'

    merge_function = lambda left_df, right_df: pd.merge(left_df, right_df, on=merge_key, how='inner')
    final_bureau_features = reduce(merge_function, dfs_to_merge)

    ids_for_agg = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]
    final_bureau_balance_features_with_curr = pd.merge(left=ids_for_agg, right=final_bureau_balance_features, how='inner')
    all_expected_status_categories = ['0', '1', '2', '3', '4', '5', 'C', 'X'] # These are the raw STATUS values
    for cat_col_name in all_expected_status_categories:
        if cat_col_name not in final_bureau_balance_features_with_curr.columns:
            final_bureau_balance_features_with_curr[cat_col_name] = 0
    final_bureau_balance_features_with_curr_mean = final_bureau_balance_features_with_curr.groupby('SK_ID_CURR')[['MONTHS_LOAN_DURATION', 'DPD_FLAG', '1', '2', '3', '4', '5', 'MEAN_DAYS_PAST_DUE', 'YEAR_LOAN_DURATION']].mean().reset_index()
    rename_mean_dict = {col: 'MEAN_' + str(col) for col in final_bureau_balance_features_with_curr_mean.columns if col != 'SK_ID_CURR'}
    final_bureau_balance_features_with_curr_mean.rename(columns=rename_mean_dict, inplace=True)



    all_expected_status_categories = ['STATUS_0','STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5', 'STATUS_C','STATUS_X'] # These are the raw STATUS values
    for cat_col_name in all_expected_status_categories:
        if cat_col_name not in final_bureau_balance_features_with_curr.columns:
            final_bureau_balance_features_with_curr[cat_col_name] = 0

    final_bureau_balance_features_with_curr_sum = final_bureau_balance_features_with_curr.groupby('SK_ID_CURR')[['STATUS_0','STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5', 'STATUS_C','STATUS_X', 'DPD_FLAG', '1', '2', '3', '4', '5', 'LOAN_TYPE_Long Term', 'LOAN_TYPE_Short Term']].sum().reset_index()
    rename_sum_dict = {col: 'SUM_' + str(col) for col in final_bureau_balance_features_with_curr_sum.columns if col != 'SK_ID_CURR'}
    final_bureau_balance_features_with_curr_sum.rename(columns=rename_sum_dict, inplace=True)

    final_bureau_balance_features_with_curr_agg = pd.merge(left=final_bureau_balance_features_with_curr_mean, right=final_bureau_balance_features_with_curr_sum, how='inner', on='SK_ID_CURR')
    features_bureau_bureau_balance = pd.merge(left=final_bureau_features, right=final_bureau_balance_features_with_curr_agg, how='inner', on='SK_ID_CURR')

    application_train_with_bureau_and_bureau_balance = pd.merge(left=current_app, right=features_bureau_bureau_balance, how='left', on='SK_ID_CURR')
    """application_train_with_bureau_and_bureau_balance[['Active', 'Bad debt', 'Closed', 'Sold',
       'CREDIT_TYPE_Another type of loan', 'CREDIT_TYPE_Car loan',
       'CREDIT_TYPE_Cash loan (non-earmarked)', 'CREDIT_TYPE_Consumer credit',
       'CREDIT_TYPE_Credit card', 'CREDIT_TYPE_Interbank credit',
       'CREDIT_TYPE_Loan for business development',
       'CREDIT_TYPE_Loan for purchase of shares (margin lending)',
       'CREDIT_TYPE_Loan for the purchase of equipment',
       'CREDIT_TYPE_Loan for working capital replenishment',
       'CREDIT_TYPE_Microloan', 'CREDIT_TYPE_Mobile operator loan',
       'CREDIT_TYPE_Mortgage', 'CREDIT_TYPE_Real estate loan',
       'CREDIT_TYPE_Unknown type of loan', 'AMT_CREDIT_MAX_OVERDUE',
       'AMT_CREDIT_SUM_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM_DEBT',
       'AMT_CREDIT_SUM', 'PROPORTION_REPAID', 'LOAN_ACTIVE_PAST_DEADLINE',
       'CREDIT_DAY_OVERDUE', 'MEAN_MONTHS_LOAN_DURATION', 'MEAN_DPD_FLAG',
       'MEAN_1', 'MEAN_2', 'MEAN_3', 'MEAN_4', 'MEAN_5',
       'MEAN_MEAN_DAYS_PAST_DUE', 'MEAN_YEAR_LOAN_DURATION', 'SUM_STATUS_0',
       'SUM_STATUS_1', 'SUM_STATUS_2', 'SUM_STATUS_3', 'SUM_STATUS_4',
       'SUM_STATUS_5', 'SUM_STATUS_C', 'SUM_STATUS_X', 'SUM_DPD_FLAG', 'SUM_1',
       'SUM_2', 'SUM_3', 'SUM_4', 'SUM_5', 'SUM_LOAN_TYPE_Long Term',
       'SUM_LOAN_TYPE_Short Term']].fillna(value=0, inplace=True)
    cols_to_fill = ['Active', 'Bad debt', 'Closed', 'Sold',
       'CREDIT_TYPE_Another type of loan', 'CREDIT_TYPE_Car loan',
       'CREDIT_TYPE_Cash loan (non-earmarked)', 'CREDIT_TYPE_Consumer credit',
       'CREDIT_TYPE_Credit card', 'CREDIT_TYPE_Interbank credit',
       'CREDIT_TYPE_Loan for business development',
       'CREDIT_TYPE_Loan for purchase of shares (margin lending)',
       'CREDIT_TYPE_Loan for the purchase of equipment',
       'CREDIT_TYPE_Loan for working capital replenishment',
       'CREDIT_TYPE_Microloan', 'CREDIT_TYPE_Mobile operator loan',
       'CREDIT_TYPE_Mortgage', 'CREDIT_TYPE_Real estate loan',
       'CREDIT_TYPE_Unknown type of loan', 'AMT_CREDIT_MAX_OVERDUE',
       'AMT_CREDIT_SUM_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM_DEBT',
       'AMT_CREDIT_SUM', 'PROPORTION_REPAID', 'LOAN_ACTIVE_PAST_DEADLINE',
       'CREDIT_DAY_OVERDUE', 'MEAN_MONTHS_LOAN_DURATION', 'MEAN_DPD_FLAG',
       'MEAN_1', 'MEAN_2', 'MEAN_3', 'MEAN_4', 'MEAN_5',
       'MEAN_MEAN_DAYS_PAST_DUE', 'MEAN_YEAR_LOAN_DURATION', 'SUM_STATUS_0',
       'SUM_STATUS_1', 'SUM_STATUS_2', 'SUM_STATUS_3', 'SUM_STATUS_4',
       'SUM_STATUS_5', 'SUM_STATUS_C', 'SUM_STATUS_X', 'SUM_DPD_FLAG', 'SUM_1',
       'SUM_2', 'SUM_3', 'SUM_4', 'SUM_5', 'SUM_LOAN_TYPE_Long Term',
       'SUM_LOAN_TYPE_Short Term']
    application_train_with_bureau_and_bureau_balance[cols_to_fill] = application_train_with_bureau_and_bureau_balance[cols_to_fill].fillna(value=0)"""

    
    # Création des features à l'aide des deux fonctions d'aggrégation
    num_features_with_function_bureau = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
    cat_features_with_function_bureau = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')

    num_features_with_function_bureau_balance = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
    cat_features_with_function_bureau_balance = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')

    num_and_cat_features_with_function_bureau_balance = pd.merge(num_features_with_function_bureau_balance, cat_features_with_function_bureau_balance, how='inner', on='SK_ID_BUREAU')
    ids_for_agg = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]
    num_and_cat_features_with_function_bureau_balance_with_curr = pd.merge(left=ids_for_agg, right=num_and_cat_features_with_function_bureau_balance, how='inner').sort_values(by='SK_ID_BUREAU', ascending=True)
    num_and_cat_features_with_function_bureau_balance_with_curr_agg = agg_numeric(num_and_cat_features_with_function_bureau_balance_with_curr.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')
    manual_final_bureau_balance_features_with_curr_agg = agg_numeric(final_bureau_balance_features_with_curr.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client_bureau_balance')

    dfs_to_merge = [num_features_with_function_bureau,
    cat_features_with_function_bureau,
    final_bureau_features,
    num_and_cat_features_with_function_bureau_balance_with_curr_agg,
    manual_final_bureau_balance_features_with_curr_agg]

    merge_key = 'SK_ID_CURR'

    merge_function = lambda left_df, right_df: pd.merge(left_df, right_df, on=merge_key, how='left')
    features_manual_and_func_from_first_three_without_app_train = reduce(merge_function, dfs_to_merge)

    features_manual_and_func_from_first_three_with_app_train = pd.merge(left=current_app, right=features_manual_and_func_from_first_three_without_app_train, how='left', on='SK_ID_CURR')

    # Aggrégation à l'aide des fonctions de previous_application, POS_CASH_balance, installments_payments et credit_card_balance
    previous_application_num_agg_SK_ID_CURR = agg_numeric(previous_application.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='previous_application')
    previous_application_cat_agg_SK_ID_CURR = count_categorical(previous_application.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='previous_application')

    POS_CASH_balance_num_agg_SK_ID_CURR = agg_numeric(POS_CASH_balance.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='POS_CASH_balance')
    POS_CASH_balance_cat_agg_SK_ID_CURR = count_categorical(POS_CASH_balance.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='POS_CASH_balance')

    credit_card_balance_num_agg_SK_ID_CURR = agg_numeric(credit_card_balance.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='credit_card_balance')
    credit_card_balance_cat_agg_SK_ID_CURR = count_categorical(credit_card_balance.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='credit_card_balance')

    installments_payments_num_agg_SK_ID_CURR = agg_numeric(installments_payments.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='credit_card_balance')

    dfs_to_merge = [installments_payments_num_agg_SK_ID_CURR, 
                previous_application_num_agg_SK_ID_CURR, previous_application_cat_agg_SK_ID_CURR, 
                POS_CASH_balance_num_agg_SK_ID_CURR, POS_CASH_balance_cat_agg_SK_ID_CURR, 
                credit_card_balance_num_agg_SK_ID_CURR, credit_card_balance_cat_agg_SK_ID_CURR]
    
    merge_key = 'SK_ID_CURR'

    merge_function = lambda left_df, right_df: pd.merge(left_df, right_df, on=merge_key, how='left')
    features_func_from_last_four = reduce(merge_function, dfs_to_merge)
    full_data = pd.merge(left=features_manual_and_func_from_first_three_with_app_train, right=features_func_from_last_four, how='left', on='SK_ID_CURR')
    print("\nDataset was properly aggregated.")

    return full_data

# @profile # Ensure commented out
def predicting_scores(full_data):
    """
    Uses the preprocessed data to predict wether or not a client is likely to repay the loan they applied for.

    Parameters
    --------
        full_data (dataframe):
            The DataFrame containing all of the features the scoring model needs

    Return
    --------
        client_with_scores (dataframe) containing all of the SK_ID_CURR and the score predicted for each
        likely_to_repay(dataframe) containing all of the SK_ID_CURR with a score strictly below the optimal threshold,
        not_likely_to_repay(dataframe) containing all of the SK_ID_CURR with a score equal or above the optimal threshold

    """
    print("--- Inside predicting_scores (Per-Request Model Load Version with Path Debugging) ---")
    # Use print for high visibility during this debug phase
    print("--- PREDICTING_SCORES_ENTRY_POINT_V2 ---") # New unique marker

      # --- MODIFICATION: Force absolute MLflow tracking URI ---
    absolute_tracking_uri = "file:///app/mlruns" # Note the three slashes for absolute path
    print(f"[PREDICT_SCORES_DEBUG] Current MLflow tracking URI before forcing: {mlflow.get_tracking_uri()}")
    print(f"[PREDICT_SCORES_DEBUG] Forcing MLflow tracking URI to: {absolute_tracking_uri}")
    try:
        mlflow.set_tracking_uri(absolute_tracking_uri)
        print(f"[PREDICT_SCORES_DEBUG] MLflow tracking URI after forcing: {mlflow.get_tracking_uri()}")
    except Exception as e_set_uri:
        print(f"[PREDICT_SCORES_DEBUG] ERROR setting tracking URI to absolute path: {e_set_uri}")
        # Depending on severity, you might want to raise this error or try to continue
    # --- END MODIFICATION ---

    run_id_of_the_pipeline = "dc6fb5c5f64e4a90b08ca76d9b962765"
    pipeline_artifact_path = "full_lgbm_pipeline" # This name is confirmed correct

    # --- NEW: Construct full model path directly ---
    pipeline_model_uri_to_use = None # Initialize
    try:
        run = mlflow.get_run(run_id_of_the_pipeline)
        base_artifact_uri_for_run = run.info.artifact_uri
        print(f"[DEBUG] Base artifact URI for run '{run_id_of_the_pipeline}': {base_artifact_uri_for_run}")

        local_base_artifact_path = None # Initialize
        if base_artifact_uri_for_run.startswith("file://"):
            local_base_artifact_path = base_artifact_uri_for_run[len("file://"):]
        elif base_artifact_uri_for_run == "artifacts": # Check for this specific case
            experiment_id = run.info.experiment_id
            # Assuming tracking URI root is /app/mlruns.
            # For more robustness, one could parse mlflow.get_tracking_uri()
            tracking_uri_root = "/app/mlruns" # Based on your forced URI
            absolute_run_path = os.path.join(tracking_uri_root, experiment_id, run_id_of_the_pipeline)
            local_base_artifact_path = os.path.join(absolute_run_path, base_artifact_uri_for_run) # This is now .../RUN_ID/artifacts
            print(f"[DEBUG] Reconstructed local_base_artifact_path from relative 'artifacts': {local_base_artifact_path}")
        else:
            print(f"[WARNING] Unexpected base_artifact_uri_for_run format: {base_artifact_uri_for_run}. Assuming it's a direct local path.")
            local_base_artifact_path = base_artifact_uri_for_run # Fallback

        if local_base_artifact_path is None:
             raise ValueError(f"Could not determine local_base_artifact_path from base_artifact_uri_for_run: {base_artifact_uri_for_run}")

        # Now, pipeline_artifact_path_in_run_artifacts is the name of the model *folder* INSIDE local_base_artifact_path
        direct_model_path = os.path.join(local_base_artifact_path, pipeline_artifact_path) # Use the correct variable name here
        print(f"[DEBUG] Constructed direct model path for loading: {direct_model_path}")
        
        if not os.path.exists(direct_model_path):
            print(f"[CRITICAL_DEBUG] The constructed direct model path DOES NOT EXIST: {direct_model_path}")
            if os.path.exists(local_base_artifact_path):
                 print(f"[CRITICAL_DEBUG] Contents of parent artifact dir '{local_base_artifact_path}': {os.listdir(local_base_artifact_path)}")
            else:
                print(f"[CRITICAL_DEBUG] Parent artifact dir '{local_base_artifact_path}' also does not exist.")
            raise FileNotFoundError(f"Manually constructed model path not found: {direct_model_path}")

        pipeline_model_uri_to_use = direct_model_path
        print(f"[DEBUG] Using direct path for model loading: {pipeline_model_uri_to_use}")

    except Exception as e_get_run:
        print(f"[DEBUG] ERROR trying to get run info or construct direct model path: {e_get_run}")
        traceback.print_exc()
        raise
    # --- END NEW ---

    print(f"[PREDICT_SCORES_DEBUG] Current MLflow tracking URI: {mlflow.get_tracking_uri()}")

    experiment_id_for_path = "448441841985485771"
    base_dir_to_check = "/app"


    # --- Get Model Info and Signature ---
    expected_input_columns = None # Initialize
    try:
        print(f"Fetching model info for URI: {pipeline_model_uri_to_use}") # This is your original print
        model_info = mlflow.models.get_model_info(pipeline_model_uri_to_use)
        if model_info.signature is not None and model_info.signature.inputs is not None:
            input_schema_obj = model_info.signature.inputs
            if hasattr(input_schema_obj, 'input_names') and callable(getattr(input_schema_obj, 'input_names')):
                expected_input_columns = input_schema_obj.input_names()
                print("\nExpected input columns for the pipeline (from signature via input_names()):")
            elif hasattr(input_schema_obj, 'inputs') and isinstance(input_schema_obj.inputs, list):
                expected_input_columns = [col_spec.name for col_spec in input_schema_obj.inputs]
                print("\nExpected input columns for the pipeline (from signature via iterating inputs):")
            else:
                print("\nCould not extract column names from input schema. Schema details:")
                print(str(input_schema_obj)[:1000])
                # expected_input_columns = None # Already initialized
        else:
            print("\nNo input signature or inputs found in model_info.")
            # expected_input_columns = None # Already initialized
    except AttributeError as ae:
        print(f"AttributeError while processing MLflow signature: {ae}")
        # expected_input_columns = None # Already initialized
    except Exception as e: # Catching a broader exception for model_info fetching
        print(f"Error fetching model info or signature from MLflow: {e}")
        # expected_input_columns = None # Already initialized

    if expected_input_columns is None:
        print("WARNING: Could not load expected_input_columns from MLflow signature. Attempting to load from 'pipeline_input_columns.txt'")
        try:
            # Inside Docker, paths are relative to WORKDIR unless absolute
            # WORKDIR is /app, so 'pipeline_input_columns.txt' should be /app/pipeline_input_columns.txt
            column_file_path = 'pipeline_input_columns.txt'
            print(f"[DEBUG] Attempting to open column file at: {os.path.abspath(column_file_path)}")
            with open(column_file_path, 'r') as f:
                loaded_cols = [line.strip() for line in f if line.strip()]
            if not loaded_cols:
                raise ValueError("pipeline_input_columns.txt is empty or contains no valid column names.")
            expected_input_columns = loaded_cols
            print(f"Successfully loaded {len(expected_input_columns)} expected input columns from {column_file_path}")
        except FileNotFoundError:
            print(f"ERROR: {column_file_path} not found (resolved to {os.path.abspath(column_file_path)}). This file is required as a fallback if MLflow signature fails.")
            raise
        except ValueError as ve:
            print(f"ERROR loading from {column_file_path}: {ve}")
            raise

    if expected_input_columns is None:
        print("CRITICAL ERROR: expected_input_columns is still None after all loading attempts. Cannot proceed.")
        raise ValueError("Failed to obtain expected input columns for the model.")
    # expected_columns_set = set(expected_input_columns) # This can be defined before column alignment

    # --- Model Loading with Debug ---
    try:
        print(f"[DEBUG] About to call mlflow.sklearn.load_model('{pipeline_model_uri_to_use}')")
        loaded_pipeline = mlflow.sklearn.load_model(pipeline_model_uri_to_use)
        print(f"\n[DEBUG] Pipeline loaded successfully via mlflow.sklearn.load_model. Type: {type(loaded_pipeline)}")
    except Exception as e_load_model:
        print(f"[DEBUG] ERROR during mlflow.sklearn.load_model: {e_load_model}")
        traceback_str = traceback.format_exc()
        print(f"[DEBUG] Traceback for load_model error:\n{traceback_str}")
        raise # Re-raise the error
    # --- End of Model Loading with Debug ---

    # --- Original processing & prediction logic (ACTIVE) ---
    print("\n(Original Log) Pipeline loaded successfully.") # Your original log line
    previously_fitted_scaler = loaded_pipeline.named_steps['standardscaler']
    print("\n(Original Log) Scaler object extracted from the pipeline.")

    print("Sanitizing full_data column names...")
    original_train_cols = full_data.columns.tolist()
    # Need to define sanitize_lgbm_colname or ensure it's imported if it's elsewhere
    new_sanitized_cols = [sanitize_lgbm_colname(col) for col in original_train_cols]
    full_data.columns = new_sanitized_cols
    print("Column names sanitized")

    if full_data.empty or 'SK_ID_CURR' not in full_data.columns:
        print("ERROR: full_data is empty or SK_ID_CURR is missing before prediction alignment. Cannot proceed.")
        raise ValueError("full_data is empty or missing SK_ID_CURR after prepare_input_data.")
    SK_ID_CURR_current_app_batch = full_data["SK_ID_CURR"].copy()

    current_columns = set(full_data.columns)
    expected_columns_set = set(expected_input_columns)
    columns_to_drop = list(current_columns - expected_columns_set)
    columns_to_drop_existing = [col for col in columns_to_drop if col in full_data.columns]
    if columns_to_drop_existing:
        full_data.drop(columns=columns_to_drop_existing, inplace=True)
    print(f"Extra columns dropped: {columns_to_drop_existing if columns_to_drop_existing else 'None'}")

    columns_to_add = list(expected_columns_set - set(full_data.columns))
    for col in columns_to_add:
        full_data[col] = np.nan
    print(f"Missing columns added with NaN values: {columns_to_add if columns_to_add else 'None'}")

    print("\nReordering columns to match the expected order...")
    # Ensure all expected_input_columns are present in full_data before reordering
    missing_for_reorder = [col for col in expected_input_columns if col not in full_data.columns]
    if missing_for_reorder:
        print(f"ERROR: Columns expected for reordering are missing from full_data: {missing_for_reorder}")
        raise ValueError(f"Cannot reorder. Columns missing: {missing_for_reorder}")
    full_data = full_data[expected_input_columns]
    print("Columns reordered.")

    if list(full_data.columns) == expected_input_columns:
        print("\nVerification successful: Columns in full_data now match expected_input_columns in name and order.")
    else:
        print("\nWARNING: Column alignment might not be perfect. Please review.")
        print(f"  Actual columns (first 10):   {list(full_data.columns)[:10]}")
        print(f"  Expected columns (first 10): {expected_input_columns[:10]}")

    print("\nMaking predictions using the loaded pipeline...")
    # Ensure full_data is not empty before predict_proba
    if full_data.empty:
        print("ERROR: full_data is empty before calling predict_proba. This should not happen if input current_app was not empty.")
        # Create empty results based on expected output structure
        # This depends on how your API needs to respond to "no data to predict"
        # For now, raising an error is clearer for debugging
        raise ValueError("full_data became empty before prediction, check alignment and input.")

    current_app_scores = loaded_pipeline.predict_proba(full_data)[:, 1]
    print("Probabilities predicted successfully.")

    current_app_scores = current_app_scores*100
    print(f"Scores (first 5): {current_app_scores[:5]}")

    optimal_threshold = 0.6336 * 100
    scores_DataFrame = pd.DataFrame({'SCORE': current_app_scores}, index=SK_ID_CURR_current_app_batch.index)
    client_with_scores = pd.concat([SK_ID_CURR_current_app_batch, scores_DataFrame], axis=1)
    likely_to_repay = client_with_scores[client_with_scores['SCORE'] < optimal_threshold]
    print(f"Likely to repay (first 5 heads if any):\n{likely_to_repay.head().to_string()}")
    not_likely_to_repay = client_with_scores[client_with_scores['SCORE'] >= optimal_threshold]
    print(f"Not likely to repay (first 5 heads if any):\n{not_likely_to_repay.head().to_string()}")

    return client_with_scores, likely_to_repay, not_likely_to_repay