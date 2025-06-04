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
def agg_numeric(df, group_var, df_name, expected_columns_after_numeric_aggregation):
    if df.empty :
        agg = pd.DataFrame(columns=expected_columns_after_numeric_aggregation)
    else :
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
def count_categorical(df, group_var, df_name, expected_columns_after_categorical_aggregation):
    if df.empty :
        categorical = pd.DataFrame(columns=expected_columns_after_categorical_aggregation)
    else :
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

    # We need to create a list of expected columns for each DataFrame because if we send to the API
    # A batch of clients where at least one DataFrame is completely empty (i.e now customers with no loan payments history)
    # We cant for example do any groupby because the columns dont exist so it will return an error
    # So we will add a check before each DataFrame treatment to make sure they're there (the columns) otherwise add them

    # LIST OF EXPECTED COLUMNS FOR EACH DATAFRAME

    initial_expected_columns_bureau = [
       'SK_ID_CURR', 'SK_ID_BUREAU', 
       'CREDIT_ACTIVE', 'CREDIT_CURRENCY',
       'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
       'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
       'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
       'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE', 'DAYS_CREDIT_UPDATE',
       'AMT_ANNUITY'
       ]
    initial_expected_columns_bureau_balance = [
        'SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS'
        ]
    initial_expected_columns_POS_CASH_balance = [
       'SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT',
       'CNT_INSTALMENT_FUTURE', 'NAME_CONTRACT_STATUS', 'SK_DPD',
       'SK_DPD_DEF'
       ]
    initial_expected_columns_installments_payments = [
       'SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_VERSION',
       'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT',
       'AMT_INSTALMENT', 'AMT_PAYMENT'
       ]
    initial_expected_columns_previous_application = [
       'SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_ANNUITY',
       'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
       'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
       'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY',
       'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
       'RATE_INTEREST_PRIVILEGED', 'NAME_CASH_LOAN_PURPOSE',
       'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 'NAME_PAYMENT_TYPE',
       'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',
       'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
       'CHANNEL_TYPE', 'SELLERPLACE_AREA', 'NAME_SELLER_INDUSTRY',
       'CNT_PAYMENT', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
       'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
       'DAYS_LAST_DUE', 'DAYS_TERMINATION', 'NFLAG_INSURED_ON_APPROVAL'
       ]
    initial_expected_columns_credit_card_balance = [
       'SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE',
       'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
       'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
       'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
       'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
       'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
       'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
       'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',
       'CNT_INSTALMENT_MATURE_CUM', 'NAME_CONTRACT_STATUS', 'SK_DPD',
       'SK_DPD_DEF'
       ]


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

    if bureau_balance.shape == (0,0) :
        bureau_balance = pd.DataFrame(columns=initial_expected_columns_bureau_balance)
    
    if bureau.shape == (0,0) :
        bureau= pd.DataFrame(columns=initial_expected_columns_bureau)
    
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
    bureau_balance_loan_duration_categorised = pd.get_dummies(
        bureau_balance_loan_duration_categorised, # This DataFrame ALREADY has 'SK_ID_BUREAU'
        columns=['LOAN_TYPE'], # This will only dummify LOAN_TYPE and not risk adding another SK_ID_BUREAU
        dtype=int
        # prefix='LOAN_TYPE' # You should add a prefix here if not already done
    )

    # Failproofnet loan type
    expected_loan_type_dummies = ['LOAN_TYPE_Long Term', 'LOAN_TYPE_Short Term']
    for col_name_to_ensure in expected_loan_type_dummies:
        if col_name_to_ensure not in bureau_balance_loan_duration_categorised.columns:
        # If the DataFrame is not empty, assign 0. 
        # If it's empty, assign an empty Series of the correct dtype.
        # This ensures the column exists for schema consistency in merges.
            if not bureau_balance_loan_duration_categorised.empty:
                bureau_balance_loan_duration_categorised[col_name_to_ensure] = 0
            else:
            # If bureau_balance_loan_duration_categorised is truly empty (no rows, no SK_ID_BUREAU),
            # creating just an empty Series might still cause issues if SK_ID_BUREAU is needed
            # for the merge into final_bureau_balance_features.
            # However, the goal here is to prevent the KeyError when these columns are *selected*.
            # If this df is empty, the subsequent inner merge for final_bureau_balance_features might
            # result in an empty df anyway, which is handled later.
                bureau_balance_loan_duration_categorised[col_name_to_ensure] = pd.Series(dtype='int')

    dfs_to_merge = [bureau_balance_loan_duration_months,
    bureau_balance_last_known_loan_status,
    bureau_balance_nombre_delais_de_paiements,
    bureau_balance_nombre_de_delais_de_paiements_par_categorie,
    bureau_balance_duree_moyenne_delais_de_paiements,
    bureau_balance_loan_duration_categorised]

    # We add a step to make sure the merge key is present to prevent errors in the reduce function

    expected_features_for_bureau_balance_loan_duration_months = ['SK_ID_BUREAU', 'MONTHS_LOAN_DURATION']
    expected_features_for_bureau_balance_last_known_loan_status = ['SK_ID_BUREAU', 'STATUS', '0', '1', '2', '3', '4', '5', 'C', 'X']
    expected_features_for_bureau_balance_nombre_delais_de_paiements = ['SK_ID_BUREAU', 'DPD_FLAG']
    expected_features_for_bureau_balance_nombre_de_delais_de_paiements_par_categorie = ['SK_ID_BUREAU', '0', '1', '2', '3', '4', '5', 'C', 'X']
    expected_features_for_bureau_balance_duree_moyenne_delais_de_paiements = ['SK_ID_BUREAU', 'MEAN_DAYS_PAST_DUE']
    expected_features_for_bureau_balance_loan_duration_categorised = ['SK_ID_BUREAU', 'YEAR_LOAN_DURATION', 'LOAN_TYPE_Long Term', 'LOAN_TYPE_Short Term']

    list_of_expected_features_per_df = [
        expected_features_for_bureau_balance_loan_duration_months, expected_features_for_bureau_balance_last_known_loan_status, 
        expected_features_for_bureau_balance_nombre_delais_de_paiements, expected_features_for_bureau_balance_nombre_de_delais_de_paiements_par_categorie, 
        expected_features_for_bureau_balance_duree_moyenne_delais_de_paiements, expected_features_for_bureau_balance_loan_duration_categorised
        ]

    validated_dfs_to_merge = []

    for df_unchecked, expected_col_list in zip(dfs_to_merge, list_of_expected_features_per_df): #utiliser zip plutot que if and sinon ca va pas faire correspondre correctement les deux conditions par exemple ça va mettre df 2 avec liste de colonne 1
        if df_unchecked.empty :
            df_corrected = pd.DataFrame(columns=expected_col_list)
            validated_dfs_to_merge.append(df_corrected)
        else :
            validated_dfs_to_merge.append(df_unchecked)

    merge_key = 'SK_ID_BUREAU'

    merge_function = lambda left_df, right_df: pd.merge(left_df, right_df, on=merge_key, how='inner')
    final_bureau_balance_features = reduce(merge_function, validated_dfs_to_merge)
    #final_bureau_balance_features = final_bureau_balance_features.drop(columns=['0', 'C', 'X'])


    bureau_for_credit_status = bureau.copy()
    bureau_number_of_type_of_credits = bureau.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].value_counts().unstack(fill_value=0).reset_index()
    bureau_number_of_type_of_credits.columns.name = None

    bureau_for_encoding_credit_types = bureau[['SK_ID_CURR', 'CREDIT_TYPE']]
    bureau_credit_type_encoded = pd.get_dummies(
        bureau_for_encoding_credit_types, 
        columns=['CREDIT_TYPE'], 
        dtype=int
        )
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

    # List of expected columns after aggregation to use in case the input DataFrame is empty

    bureau_expected_columns_after_numeric_aggregation = [
       'SK_ID_CURR', 'bureau_DAYS_CREDIT_count', 'bureau_DAYS_CREDIT_mean',
       'bureau_DAYS_CREDIT_max', 'bureau_DAYS_CREDIT_min',
       'bureau_DAYS_CREDIT_sum', 'bureau_CREDIT_DAY_OVERDUE_count',
       'bureau_CREDIT_DAY_OVERDUE_mean', 'bureau_CREDIT_DAY_OVERDUE_max',
       'bureau_CREDIT_DAY_OVERDUE_min', 'bureau_CREDIT_DAY_OVERDUE_sum',
       'bureau_DAYS_CREDIT_ENDDATE_count', 'bureau_DAYS_CREDIT_ENDDATE_mean',
       'bureau_DAYS_CREDIT_ENDDATE_max', 'bureau_DAYS_CREDIT_ENDDATE_min',
       'bureau_DAYS_CREDIT_ENDDATE_sum', 'bureau_DAYS_ENDDATE_FACT_count',
       'bureau_DAYS_ENDDATE_FACT_mean', 'bureau_DAYS_ENDDATE_FACT_max',
       'bureau_DAYS_ENDDATE_FACT_min', 'bureau_DAYS_ENDDATE_FACT_sum',
       'bureau_AMT_CREDIT_MAX_OVERDUE_count',
       'bureau_AMT_CREDIT_MAX_OVERDUE_mean',
       'bureau_AMT_CREDIT_MAX_OVERDUE_max',
       'bureau_AMT_CREDIT_MAX_OVERDUE_min',
       'bureau_AMT_CREDIT_MAX_OVERDUE_sum', 'bureau_CNT_CREDIT_PROLONG_count',
       'bureau_CNT_CREDIT_PROLONG_mean', 'bureau_CNT_CREDIT_PROLONG_max',
       'bureau_CNT_CREDIT_PROLONG_min', 'bureau_CNT_CREDIT_PROLONG_sum',
       'bureau_AMT_CREDIT_SUM_count', 'bureau_AMT_CREDIT_SUM_mean',
       'bureau_AMT_CREDIT_SUM_max', 'bureau_AMT_CREDIT_SUM_min',
       'bureau_AMT_CREDIT_SUM_sum', 'bureau_AMT_CREDIT_SUM_DEBT_count',
       'bureau_AMT_CREDIT_SUM_DEBT_mean', 'bureau_AMT_CREDIT_SUM_DEBT_max',
       'bureau_AMT_CREDIT_SUM_DEBT_min', 'bureau_AMT_CREDIT_SUM_DEBT_sum',
       'bureau_AMT_CREDIT_SUM_LIMIT_count', 'bureau_AMT_CREDIT_SUM_LIMIT_mean',
       'bureau_AMT_CREDIT_SUM_LIMIT_max', 'bureau_AMT_CREDIT_SUM_LIMIT_min',
       'bureau_AMT_CREDIT_SUM_LIMIT_sum',
       'bureau_AMT_CREDIT_SUM_OVERDUE_count',
       'bureau_AMT_CREDIT_SUM_OVERDUE_mean',
       'bureau_AMT_CREDIT_SUM_OVERDUE_max',
       'bureau_AMT_CREDIT_SUM_OVERDUE_min',
       'bureau_AMT_CREDIT_SUM_OVERDUE_sum', 'bureau_DAYS_CREDIT_UPDATE_count',
       'bureau_DAYS_CREDIT_UPDATE_mean', 'bureau_DAYS_CREDIT_UPDATE_max',
       'bureau_DAYS_CREDIT_UPDATE_min', 'bureau_DAYS_CREDIT_UPDATE_sum',
       'bureau_AMT_ANNUITY_count', 'bureau_AMT_ANNUITY_mean',
       'bureau_AMT_ANNUITY_max', 'bureau_AMT_ANNUITY_min',
       'bureau_AMT_ANNUITY_sum'
       ]
    
    bureau_expected_columns_after_categorical_aggregation = [
       'bureau_CREDIT_ACTIVE_Active_count',
       'bureau_CREDIT_ACTIVE_Active_count_norm',
       'bureau_CREDIT_ACTIVE_Bad debt_count',
       'bureau_CREDIT_ACTIVE_Bad debt_count_norm',
       'bureau_CREDIT_ACTIVE_Closed_count',
       'bureau_CREDIT_ACTIVE_Closed_count_norm',
       'bureau_CREDIT_ACTIVE_Sold_count',
       'bureau_CREDIT_ACTIVE_Sold_count_norm',
       'bureau_CREDIT_CURRENCY_currency 1_count',
       'bureau_CREDIT_CURRENCY_currency 1_count_norm',
       'bureau_CREDIT_CURRENCY_currency 2_count',
       'bureau_CREDIT_CURRENCY_currency 2_count_norm',
       'bureau_CREDIT_CURRENCY_currency 3_count',
       'bureau_CREDIT_CURRENCY_currency 3_count_norm',
       'bureau_CREDIT_CURRENCY_currency 4_count',
       'bureau_CREDIT_CURRENCY_currency 4_count_norm',
       'bureau_CREDIT_TYPE_Another type of loan_count',
       'bureau_CREDIT_TYPE_Another type of loan_count_norm',
       'bureau_CREDIT_TYPE_Car loan_count',
       'bureau_CREDIT_TYPE_Car loan_count_norm',
       'bureau_CREDIT_TYPE_Cash loan (non-earmarked)_count',
       'bureau_CREDIT_TYPE_Cash loan (non-earmarked)_count_norm',
       'bureau_CREDIT_TYPE_Consumer credit_count',
       'bureau_CREDIT_TYPE_Consumer credit_count_norm',
       'bureau_CREDIT_TYPE_Credit card_count',
       'bureau_CREDIT_TYPE_Credit card_count_norm',
       'bureau_CREDIT_TYPE_Interbank credit_count',
       'bureau_CREDIT_TYPE_Interbank credit_count_norm',
       'bureau_CREDIT_TYPE_Loan for business development_count',
       'bureau_CREDIT_TYPE_Loan for business development_count_norm',
       'bureau_CREDIT_TYPE_Loan for purchase of shares (margin lending)_count',
       'bureau_CREDIT_TYPE_Loan for purchase of shares (margin lending)_count_norm',
       'bureau_CREDIT_TYPE_Loan for the purchase of equipment_count',
       'bureau_CREDIT_TYPE_Loan for the purchase of equipment_count_norm',
       'bureau_CREDIT_TYPE_Loan for working capital replenishment_count',
       'bureau_CREDIT_TYPE_Loan for working capital replenishment_count_norm',
       'bureau_CREDIT_TYPE_Microloan_count',
       'bureau_CREDIT_TYPE_Microloan_count_norm',
       'bureau_CREDIT_TYPE_Mobile operator loan_count',
       'bureau_CREDIT_TYPE_Mobile operator loan_count_norm',
       'bureau_CREDIT_TYPE_Mortgage_count',
       'bureau_CREDIT_TYPE_Mortgage_count_norm',
       'bureau_CREDIT_TYPE_Real estate loan_count',
       'bureau_CREDIT_TYPE_Real estate loan_count_norm',
       'bureau_CREDIT_TYPE_Unknown type of loan_count',
       'bureau_CREDIT_TYPE_Unknown type of loan_count_norm'
       ]
    
    bureau_balance_expected_columns_after_numeric_aggregation = [
       'SK_ID_BUREAU', 'bureau_balance_MONTHS_BALANCE_count',
       'bureau_balance_MONTHS_BALANCE_mean',
       'bureau_balance_MONTHS_BALANCE_max',
       'bureau_balance_MONTHS_BALANCE_min',
       'bureau_balance_MONTHS_BALANCE_sum'
       ]
    
    bureau_balance_expected_columns_after_categorical_aggregation = [
       'bureau_balance_STATUS_0_count', 'bureau_balance_STATUS_0_count_norm',
       'bureau_balance_STATUS_1_count', 'bureau_balance_STATUS_1_count_norm',
       'bureau_balance_STATUS_2_count', 'bureau_balance_STATUS_2_count_norm',
       'bureau_balance_STATUS_3_count', 'bureau_balance_STATUS_3_count_norm',
       'bureau_balance_STATUS_4_count', 'bureau_balance_STATUS_4_count_norm',
       'bureau_balance_STATUS_5_count', 'bureau_balance_STATUS_5_count_norm',
       'bureau_balance_STATUS_C_count', 'bureau_balance_STATUS_C_count_norm',
       'bureau_balance_STATUS_X_count', 'bureau_balance_STATUS_X_count_norm'
       ]
    
    # Création des features à l'aide des deux fonctions d'aggrégation
    num_features_with_function_bureau = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau', expected_columns_after_numeric_aggregation=bureau_expected_columns_after_numeric_aggregation)
    cat_features_with_function_bureau = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau', expected_columns_after_categorical_aggregation=bureau_expected_columns_after_categorical_aggregation)

    num_features_with_function_bureau_balance = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance', expected_columns_after_numeric_aggregation=bureau_balance_expected_columns_after_numeric_aggregation)
    cat_features_with_function_bureau_balance = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance', expected_columns_after_categorical_aggregation=bureau_balance_expected_columns_after_categorical_aggregation)

    num_and_cat_features_with_function_bureau_balance = pd.merge(num_features_with_function_bureau_balance, cat_features_with_function_bureau_balance, how='inner', on='SK_ID_BUREAU')
    ids_for_agg = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]
    num_and_cat_features_with_function_bureau_balance_with_curr = pd.merge(left=ids_for_agg, right=num_and_cat_features_with_function_bureau_balance, how='inner').sort_values(by='SK_ID_BUREAU', ascending=True)
    
    num_and_cat_features_with_function_bureau_balance_with_curr_expected_columns_after_numeric_aggregation = ['SK_ID_CURR', 'client_bureau_balance_MONTHS_BALANCE_count_count', 'client_bureau_balance_MONTHS_BALANCE_count_mean', 'client_bureau_balance_MONTHS_BALANCE_count_max', 'client_bureau_balance_MONTHS_BALANCE_count_min', 'client_bureau_balance_MONTHS_BALANCE_count_sum', 'client_bureau_balance_MONTHS_BALANCE_mean_count', 'client_bureau_balance_MONTHS_BALANCE_mean_mean', 'client_bureau_balance_MONTHS_BALANCE_mean_max', 'client_bureau_balance_MONTHS_BALANCE_mean_min', 'client_bureau_balance_MONTHS_BALANCE_mean_sum', 'client_bureau_balance_MONTHS_BALANCE_max_count', 'client_bureau_balance_MONTHS_BALANCE_max_mean', 'client_bureau_balance_MONTHS_BALANCE_max_max', 'client_bureau_balance_MONTHS_BALANCE_max_min', 'client_bureau_balance_MONTHS_BALANCE_max_sum', 'client_bureau_balance_MONTHS_BALANCE_min_count', 'client_bureau_balance_MONTHS_BALANCE_min_mean', 'client_bureau_balance_MONTHS_BALANCE_min_max', 'client_bureau_balance_MONTHS_BALANCE_min_min', 'client_bureau_balance_MONTHS_BALANCE_min_sum', 'client_bureau_balance_MONTHS_BALANCE_sum_count', 'client_bureau_balance_MONTHS_BALANCE_sum_mean', 'client_bureau_balance_MONTHS_BALANCE_sum_max', 'client_bureau_balance_MONTHS_BALANCE_sum_min', 'client_bureau_balance_MONTHS_BALANCE_sum_sum', 'client_bureau_balance_STATUS_0_count_count', 'client_bureau_balance_STATUS_0_count_mean', 'client_bureau_balance_STATUS_0_count_max', 'client_bureau_balance_STATUS_0_count_min', 'client_bureau_balance_STATUS_0_count_sum', 'client_bureau_balance_STATUS_0_count_norm_count', 'client_bureau_balance_STATUS_0_count_norm_mean', 'client_bureau_balance_STATUS_0_count_norm_max', 'client_bureau_balance_STATUS_0_count_norm_min', 'client_bureau_balance_STATUS_0_count_norm_sum', 'client_bureau_balance_STATUS_1_count_count', 'client_bureau_balance_STATUS_1_count_mean', 'client_bureau_balance_STATUS_1_count_max', 'client_bureau_balance_STATUS_1_count_min', 'client_bureau_balance_STATUS_1_count_sum', 'client_bureau_balance_STATUS_1_count_norm_count', 'client_bureau_balance_STATUS_1_count_norm_mean', 'client_bureau_balance_STATUS_1_count_norm_max', 'client_bureau_balance_STATUS_1_count_norm_min', 'client_bureau_balance_STATUS_1_count_norm_sum', 'client_bureau_balance_STATUS_2_count_count', 'client_bureau_balance_STATUS_2_count_mean', 'client_bureau_balance_STATUS_2_count_max', 'client_bureau_balance_STATUS_2_count_min', 'client_bureau_balance_STATUS_2_count_sum', 'client_bureau_balance_STATUS_2_count_norm_count', 'client_bureau_balance_STATUS_2_count_norm_mean', 'client_bureau_balance_STATUS_2_count_norm_max', 'client_bureau_balance_STATUS_2_count_norm_min', 'client_bureau_balance_STATUS_2_count_norm_sum', 'client_bureau_balance_STATUS_3_count_count', 'client_bureau_balance_STATUS_3_count_mean', 'client_bureau_balance_STATUS_3_count_max', 'client_bureau_balance_STATUS_3_count_min', 'client_bureau_balance_STATUS_3_count_sum', 'client_bureau_balance_STATUS_3_count_norm_count', 'client_bureau_balance_STATUS_3_count_norm_mean', 'client_bureau_balance_STATUS_3_count_norm_max', 'client_bureau_balance_STATUS_3_count_norm_min', 'client_bureau_balance_STATUS_3_count_norm_sum', 'client_bureau_balance_STATUS_4_count_count', 'client_bureau_balance_STATUS_4_count_mean', 'client_bureau_balance_STATUS_4_count_max', 'client_bureau_balance_STATUS_4_count_min', 'client_bureau_balance_STATUS_4_count_sum', 'client_bureau_balance_STATUS_4_count_norm_count', 'client_bureau_balance_STATUS_4_count_norm_mean', 'client_bureau_balance_STATUS_4_count_norm_max', 'client_bureau_balance_STATUS_4_count_norm_min', 'client_bureau_balance_STATUS_4_count_norm_sum', 'client_bureau_balance_STATUS_5_count_count', 'client_bureau_balance_STATUS_5_count_mean', 'client_bureau_balance_STATUS_5_count_max', 'client_bureau_balance_STATUS_5_count_min', 'client_bureau_balance_STATUS_5_count_sum', 'client_bureau_balance_STATUS_5_count_norm_count', 'client_bureau_balance_STATUS_5_count_norm_mean', 'client_bureau_balance_STATUS_5_count_norm_max', 'client_bureau_balance_STATUS_5_count_norm_min', 'client_bureau_balance_STATUS_5_count_norm_sum', 'client_bureau_balance_STATUS_C_count_count', 'client_bureau_balance_STATUS_C_count_mean', 'client_bureau_balance_STATUS_C_count_max', 'client_bureau_balance_STATUS_C_count_min', 'client_bureau_balance_STATUS_C_count_sum', 'client_bureau_balance_STATUS_C_count_norm_count', 'client_bureau_balance_STATUS_C_count_norm_mean', 'client_bureau_balance_STATUS_C_count_norm_max', 'client_bureau_balance_STATUS_C_count_norm_min', 'client_bureau_balance_STATUS_C_count_norm_sum', 'client_bureau_balance_STATUS_X_count_count', 'client_bureau_balance_STATUS_X_count_mean', 'client_bureau_balance_STATUS_X_count_max', 'client_bureau_balance_STATUS_X_count_min', 'client_bureau_balance_STATUS_X_count_sum', 'client_bureau_balance_STATUS_X_count_norm_count', 'client_bureau_balance_STATUS_X_count_norm_mean', 'client_bureau_balance_STATUS_X_count_norm_max', 'client_bureau_balance_STATUS_X_count_norm_min', 'client_bureau_balance_STATUS_X_count_norm_sum']

    num_and_cat_features_with_function_bureau_balance_with_curr_agg = agg_numeric(num_and_cat_features_with_function_bureau_balance_with_curr.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client', expected_columns_after_numeric_aggregation=num_and_cat_features_with_function_bureau_balance_with_curr_expected_columns_after_numeric_aggregation)
    
    manual_final_bureau_balance_features_with_curr_expected_columns_after_numeric_aggregation = ['SK_ID_CURR', 'client_bureau_balance_MONTHS_LOAN_DURATION_count', 'client_bureau_balance_MONTHS_LOAN_DURATION_mean', 'client_bureau_balance_MONTHS_LOAN_DURATION_max', 'client_bureau_balance_MONTHS_LOAN_DURATION_min', 'client_bureau_balance_MONTHS_LOAN_DURATION_sum', 'client_bureau_balance_0_x_count', 'client_bureau_balance_0_x_mean', 'client_bureau_balance_0_x_max', 'client_bureau_balance_0_x_min', 'client_bureau_balance_0_x_sum', 'client_bureau_balance_1_x_count', 'client_bureau_balance_1_x_mean', 'client_bureau_balance_1_x_max', 'client_bureau_balance_1_x_min', 'client_bureau_balance_1_x_sum', 'client_bureau_balance_2_x_count', 'client_bureau_balance_2_x_mean', 'client_bureau_balance_2_x_max', 'client_bureau_balance_2_x_min', 'client_bureau_balance_2_x_sum', 'client_bureau_balance_3_x_count', 'client_bureau_balance_3_x_mean', 'client_bureau_balance_3_x_max', 'client_bureau_balance_3_x_min', 'client_bureau_balance_3_x_sum', 'client_bureau_balance_4_x_count', 'client_bureau_balance_4_x_mean', 'client_bureau_balance_4_x_max', 'client_bureau_balance_4_x_min', 'client_bureau_balance_4_x_sum', 'client_bureau_balance_5_x_count', 'client_bureau_balance_5_x_mean', 'client_bureau_balance_5_x_max', 'client_bureau_balance_5_x_min', 'client_bureau_balance_5_x_sum', 'client_bureau_balance_C_x_count', 'client_bureau_balance_C_x_mean', 'client_bureau_balance_C_x_max', 'client_bureau_balance_C_x_min', 'client_bureau_balance_C_x_sum', 'client_bureau_balance_X_x_count', 'client_bureau_balance_X_x_mean', 'client_bureau_balance_X_x_max', 'client_bureau_balance_X_x_min', 'client_bureau_balance_X_x_sum', 'client_bureau_balance_DPD_FLAG_count', 'client_bureau_balance_DPD_FLAG_mean', 'client_bureau_balance_DPD_FLAG_max', 'client_bureau_balance_DPD_FLAG_min', 'client_bureau_balance_DPD_FLAG_sum', 'client_bureau_balance_0_y_count', 'client_bureau_balance_0_y_mean', 'client_bureau_balance_0_y_max', 'client_bureau_balance_0_y_min', 'client_bureau_balance_0_y_sum', 'client_bureau_balance_1_y_count', 'client_bureau_balance_1_y_mean', 'client_bureau_balance_1_y_max', 'client_bureau_balance_1_y_min', 'client_bureau_balance_1_y_sum', 'client_bureau_balance_2_y_count', 'client_bureau_balance_2_y_mean', 'client_bureau_balance_2_y_max', 'client_bureau_balance_2_y_min', 'client_bureau_balance_2_y_sum', 'client_bureau_balance_3_y_count', 'client_bureau_balance_3_y_mean', 'client_bureau_balance_3_y_max', 'client_bureau_balance_3_y_min', 'client_bureau_balance_3_y_sum', 'client_bureau_balance_4_y_count', 'client_bureau_balance_4_y_mean', 'client_bureau_balance_4_y_max', 'client_bureau_balance_4_y_min', 'client_bureau_balance_4_y_sum', 'client_bureau_balance_5_y_count', 'client_bureau_balance_5_y_mean', 'client_bureau_balance_5_y_max', 'client_bureau_balance_5_y_min', 'client_bureau_balance_5_y_sum', 'client_bureau_balance_C_y_count', 'client_bureau_balance_C_y_mean', 'client_bureau_balance_C_y_max', 'client_bureau_balance_C_y_min', 'client_bureau_balance_C_y_sum', 'client_bureau_balance_X_y_count', 'client_bureau_balance_X_y_mean', 'client_bureau_balance_X_y_max', 'client_bureau_balance_X_y_min', 'client_bureau_balance_X_y_sum', 'client_bureau_balance_MEAN_DAYS_PAST_DUE_count', 'client_bureau_balance_MEAN_DAYS_PAST_DUE_mean', 'client_bureau_balance_MEAN_DAYS_PAST_DUE_max', 'client_bureau_balance_MEAN_DAYS_PAST_DUE_min', 'client_bureau_balance_MEAN_DAYS_PAST_DUE_sum', 'client_bureau_balance_YEAR_LOAN_DURATION_count', 'client_bureau_balance_YEAR_LOAN_DURATION_mean', 'client_bureau_balance_YEAR_LOAN_DURATION_max', 'client_bureau_balance_YEAR_LOAN_DURATION_min', 'client_bureau_balance_YEAR_LOAN_DURATION_sum', 'client_bureau_balance_LOAN_TYPE_Long Term_count', 'client_bureau_balance_LOAN_TYPE_Long Term_mean', 'client_bureau_balance_LOAN_TYPE_Long Term_max', 'client_bureau_balance_LOAN_TYPE_Long Term_min', 'client_bureau_balance_LOAN_TYPE_Long Term_sum', 'client_bureau_balance_LOAN_TYPE_Short Term_count', 'client_bureau_balance_LOAN_TYPE_Short Term_mean', 'client_bureau_balance_LOAN_TYPE_Short Term_max', 'client_bureau_balance_LOAN_TYPE_Short Term_min', 'client_bureau_balance_LOAN_TYPE_Short Term_sum', 'client_bureau_balance_0_count', 'client_bureau_balance_0_mean', 'client_bureau_balance_0_max', 'client_bureau_balance_0_min', 'client_bureau_balance_0_sum', 'client_bureau_balance_1_count', 'client_bureau_balance_1_mean', 'client_bureau_balance_1_max', 'client_bureau_balance_1_min', 'client_bureau_balance_1_sum', 'client_bureau_balance_2_count', 'client_bureau_balance_2_mean', 'client_bureau_balance_2_max', 'client_bureau_balance_2_min', 'client_bureau_balance_2_sum', 'client_bureau_balance_3_count', 'client_bureau_balance_3_mean', 'client_bureau_balance_3_max', 'client_bureau_balance_3_min', 'client_bureau_balance_3_sum', 'client_bureau_balance_4_count', 'client_bureau_balance_4_mean', 'client_bureau_balance_4_max', 'client_bureau_balance_4_min', 'client_bureau_balance_4_sum', 'client_bureau_balance_5_count', 'client_bureau_balance_5_mean', 'client_bureau_balance_5_max', 'client_bureau_balance_5_min', 'client_bureau_balance_5_sum', 'client_bureau_balance_C_count', 'client_bureau_balance_C_mean', 'client_bureau_balance_C_max', 'client_bureau_balance_C_min', 'client_bureau_balance_C_sum', 'client_bureau_balance_X_count', 'client_bureau_balance_X_mean', 'client_bureau_balance_X_max', 'client_bureau_balance_X_min', 'client_bureau_balance_X_sum', 'client_bureau_balance_STATUS_0_count', 'client_bureau_balance_STATUS_0_mean', 'client_bureau_balance_STATUS_0_max', 'client_bureau_balance_STATUS_0_min', 'client_bureau_balance_STATUS_0_sum', 'client_bureau_balance_STATUS_1_count', 'client_bureau_balance_STATUS_1_mean', 'client_bureau_balance_STATUS_1_max', 'client_bureau_balance_STATUS_1_min', 'client_bureau_balance_STATUS_1_sum', 'client_bureau_balance_STATUS_2_count', 'client_bureau_balance_STATUS_2_mean', 'client_bureau_balance_STATUS_2_max', 'client_bureau_balance_STATUS_2_min', 'client_bureau_balance_STATUS_2_sum', 'client_bureau_balance_STATUS_3_count', 'client_bureau_balance_STATUS_3_mean', 'client_bureau_balance_STATUS_3_max', 'client_bureau_balance_STATUS_3_min', 'client_bureau_balance_STATUS_3_sum', 'client_bureau_balance_STATUS_4_count', 'client_bureau_balance_STATUS_4_mean', 'client_bureau_balance_STATUS_4_max', 'client_bureau_balance_STATUS_4_min', 'client_bureau_balance_STATUS_4_sum', 'client_bureau_balance_STATUS_5_count', 'client_bureau_balance_STATUS_5_mean', 'client_bureau_balance_STATUS_5_max', 'client_bureau_balance_STATUS_5_min', 'client_bureau_balance_STATUS_5_sum', 'client_bureau_balance_STATUS_C_count', 'client_bureau_balance_STATUS_C_mean', 'client_bureau_balance_STATUS_C_max', 'client_bureau_balance_STATUS_C_min', 'client_bureau_balance_STATUS_C_sum', 'client_bureau_balance_STATUS_X_count', 'client_bureau_balance_STATUS_X_mean', 'client_bureau_balance_STATUS_X_max', 'client_bureau_balance_STATUS_X_min', 'client_bureau_balance_STATUS_X_sum']
    
    manual_final_bureau_balance_features_with_curr_agg = agg_numeric(final_bureau_balance_features_with_curr.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client_bureau_balance', expected_columns_after_numeric_aggregation=manual_final_bureau_balance_features_with_curr_expected_columns_after_numeric_aggregation)

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

    if previous_application.shape == (0,0) : 
        previous_application = pd.DataFrame(columns=initial_expected_columns_previous_application)
    if POS_CASH_balance.shape == (0,0) : 
        POS_CASH_balance = pd.DataFrame(columns=initial_expected_columns_POS_CASH_balance)
    if installments_payments.shape == (0,0) : 
        installments_payments = pd.DataFrame(columns=initial_expected_columns_installments_payments)
    if credit_card_balance.shape == (0,0) : 
        credit_card_balance = pd.DataFrame(columns=initial_expected_columns_credit_card_balance)

    
    # List of expected columns after aggregation in case input is empty for previous_application

    previous_application_expected_columns_after_numeric_aggregation = [
       'SK_ID_CURR', 'previous_application_AMT_ANNUITY_count',
       'previous_application_AMT_ANNUITY_mean',
       'previous_application_AMT_ANNUITY_max',
       'previous_application_AMT_ANNUITY_min',
       'previous_application_AMT_ANNUITY_sum',
       'previous_application_AMT_APPLICATION_count',
       'previous_application_AMT_APPLICATION_mean',
       'previous_application_AMT_APPLICATION_max',
       'previous_application_AMT_APPLICATION_min',
       'previous_application_AMT_APPLICATION_sum',
       'previous_application_AMT_CREDIT_count',
       'previous_application_AMT_CREDIT_mean',
       'previous_application_AMT_CREDIT_max',
       'previous_application_AMT_CREDIT_min',
       'previous_application_AMT_CREDIT_sum',
       'previous_application_AMT_DOWN_PAYMENT_count',
       'previous_application_AMT_DOWN_PAYMENT_mean',
       'previous_application_AMT_DOWN_PAYMENT_max',
       'previous_application_AMT_DOWN_PAYMENT_min',
       'previous_application_AMT_DOWN_PAYMENT_sum',
       'previous_application_AMT_GOODS_PRICE_count',
       'previous_application_AMT_GOODS_PRICE_mean',
       'previous_application_AMT_GOODS_PRICE_max',
       'previous_application_AMT_GOODS_PRICE_min',
       'previous_application_AMT_GOODS_PRICE_sum',
       'previous_application_HOUR_APPR_PROCESS_START_count',
       'previous_application_HOUR_APPR_PROCESS_START_mean',
       'previous_application_HOUR_APPR_PROCESS_START_max',
       'previous_application_HOUR_APPR_PROCESS_START_min',
       'previous_application_HOUR_APPR_PROCESS_START_sum',
       'previous_application_NFLAG_LAST_APPL_IN_DAY_count',
       'previous_application_NFLAG_LAST_APPL_IN_DAY_mean',
       'previous_application_NFLAG_LAST_APPL_IN_DAY_max',
       'previous_application_NFLAG_LAST_APPL_IN_DAY_min',
       'previous_application_NFLAG_LAST_APPL_IN_DAY_sum',
       'previous_application_RATE_DOWN_PAYMENT_count',
       'previous_application_RATE_DOWN_PAYMENT_mean',
       'previous_application_RATE_DOWN_PAYMENT_max',
       'previous_application_RATE_DOWN_PAYMENT_min',
       'previous_application_RATE_DOWN_PAYMENT_sum',
       'previous_application_RATE_INTEREST_PRIMARY_count',
       'previous_application_RATE_INTEREST_PRIMARY_mean',
       'previous_application_RATE_INTEREST_PRIMARY_max',
       'previous_application_RATE_INTEREST_PRIMARY_min',
       'previous_application_RATE_INTEREST_PRIMARY_sum',
       'previous_application_RATE_INTEREST_PRIVILEGED_count',
       'previous_application_RATE_INTEREST_PRIVILEGED_mean',
       'previous_application_RATE_INTEREST_PRIVILEGED_max',
       'previous_application_RATE_INTEREST_PRIVILEGED_min',
       'previous_application_RATE_INTEREST_PRIVILEGED_sum',
       'previous_application_DAYS_DECISION_count',
       'previous_application_DAYS_DECISION_mean',
       'previous_application_DAYS_DECISION_max',
       'previous_application_DAYS_DECISION_min',
       'previous_application_DAYS_DECISION_sum',
       'previous_application_SELLERPLACE_AREA_count',
       'previous_application_SELLERPLACE_AREA_mean',
       'previous_application_SELLERPLACE_AREA_max',
       'previous_application_SELLERPLACE_AREA_min',
       'previous_application_SELLERPLACE_AREA_sum',
       'previous_application_CNT_PAYMENT_count',
       'previous_application_CNT_PAYMENT_mean',
       'previous_application_CNT_PAYMENT_max',
       'previous_application_CNT_PAYMENT_min',
       'previous_application_CNT_PAYMENT_sum',
       'previous_application_DAYS_FIRST_DRAWING_count',
       'previous_application_DAYS_FIRST_DRAWING_mean',
       'previous_application_DAYS_FIRST_DRAWING_max',
       'previous_application_DAYS_FIRST_DRAWING_min',
       'previous_application_DAYS_FIRST_DRAWING_sum',
       'previous_application_DAYS_FIRST_DUE_count',
       'previous_application_DAYS_FIRST_DUE_mean',
       'previous_application_DAYS_FIRST_DUE_max',
       'previous_application_DAYS_FIRST_DUE_min',
       'previous_application_DAYS_FIRST_DUE_sum',
       'previous_application_DAYS_LAST_DUE_1ST_VERSION_count',
       'previous_application_DAYS_LAST_DUE_1ST_VERSION_mean',
       'previous_application_DAYS_LAST_DUE_1ST_VERSION_max',
       'previous_application_DAYS_LAST_DUE_1ST_VERSION_min',
       'previous_application_DAYS_LAST_DUE_1ST_VERSION_sum',
       'previous_application_DAYS_LAST_DUE_count',
       'previous_application_DAYS_LAST_DUE_mean',
       'previous_application_DAYS_LAST_DUE_max',
       'previous_application_DAYS_LAST_DUE_min',
       'previous_application_DAYS_LAST_DUE_sum',
       'previous_application_DAYS_TERMINATION_count',
       'previous_application_DAYS_TERMINATION_mean',
       'previous_application_DAYS_TERMINATION_max',
       'previous_application_DAYS_TERMINATION_min',
       'previous_application_DAYS_TERMINATION_sum',
       'previous_application_NFLAG_INSURED_ON_APPROVAL_count',
       'previous_application_NFLAG_INSURED_ON_APPROVAL_mean',
       'previous_application_NFLAG_INSURED_ON_APPROVAL_max',
       'previous_application_NFLAG_INSURED_ON_APPROVAL_min',
       'previous_application_NFLAG_INSURED_ON_APPROVAL_sum'
       ]
    
    previous_application_expected_columns_after_categorical_aggregation = ['previous_application_NAME_CONTRACT_TYPE_Cash loans_count', 'previous_application_NAME_CONTRACT_TYPE_Cash loans_count_norm', 'previous_application_NAME_CONTRACT_TYPE_Consumer loans_count', 'previous_application_NAME_CONTRACT_TYPE_Consumer loans_count_norm', 'previous_application_NAME_CONTRACT_TYPE_Revolving loans_count', 'previous_application_NAME_CONTRACT_TYPE_Revolving loans_count_norm', 'previous_application_NAME_CONTRACT_TYPE_XNA_count', 'previous_application_NAME_CONTRACT_TYPE_XNA_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_FRIDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_FRIDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_MONDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_MONDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_SATURDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_SATURDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_SUNDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_SUNDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_THURSDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_THURSDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_TUESDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_TUESDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_count_norm', 'previous_application_FLAG_LAST_APPL_PER_CONTRACT_N_count', 'previous_application_FLAG_LAST_APPL_PER_CONTRACT_N_count_norm', 'previous_application_FLAG_LAST_APPL_PER_CONTRACT_Y_count', 'previous_application_FLAG_LAST_APPL_PER_CONTRACT_Y_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Business development_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Business development_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a garage_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a garage_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a home_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a home_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a new car_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a new car_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a used car_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a used car_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Car repairs_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Car repairs_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Education_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Education_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Everyday expenses_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Everyday expenses_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Furniture_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Furniture_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Hobby_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Hobby_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Journey_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Journey_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Medicine_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Medicine_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Money for a third person_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Money for a third person_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Other_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Other_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Payments on other loans_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Payments on other loans_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Repairs_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Repairs_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Urgent needs_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Urgent needs_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_XAP_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_XAP_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_XNA_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_XNA_count_norm', 'previous_application_NAME_CONTRACT_STATUS_Approved_count', 'previous_application_NAME_CONTRACT_STATUS_Approved_count_norm', 'previous_application_NAME_CONTRACT_STATUS_Canceled_count', 'previous_application_NAME_CONTRACT_STATUS_Canceled_count_norm', 'previous_application_NAME_CONTRACT_STATUS_Refused_count', 'previous_application_NAME_CONTRACT_STATUS_Refused_count_norm', 'previous_application_NAME_CONTRACT_STATUS_Unused offer_count', 'previous_application_NAME_CONTRACT_STATUS_Unused offer_count_norm', 'previous_application_NAME_PAYMENT_TYPE_Cash through the bank_count', 'previous_application_NAME_PAYMENT_TYPE_Cash through the bank_count_norm', 'previous_application_NAME_PAYMENT_TYPE_Cashless from the account of the employer_count', 'previous_application_NAME_PAYMENT_TYPE_Cashless from the account of the employer_count_norm', 'previous_application_NAME_PAYMENT_TYPE_Non-cash from your account_count', 'previous_application_NAME_PAYMENT_TYPE_Non-cash from your account_count_norm', 'previous_application_NAME_PAYMENT_TYPE_XNA_count', 'previous_application_NAME_PAYMENT_TYPE_XNA_count_norm', 'previous_application_CODE_REJECT_REASON_CLIENT_count', 'previous_application_CODE_REJECT_REASON_CLIENT_count_norm', 'previous_application_CODE_REJECT_REASON_HC_count', 'previous_application_CODE_REJECT_REASON_HC_count_norm', 'previous_application_CODE_REJECT_REASON_LIMIT_count', 'previous_application_CODE_REJECT_REASON_LIMIT_count_norm', 'previous_application_CODE_REJECT_REASON_SCO_count', 'previous_application_CODE_REJECT_REASON_SCO_count_norm', 'previous_application_CODE_REJECT_REASON_SCOFR_count', 'previous_application_CODE_REJECT_REASON_SCOFR_count_norm', 'previous_application_CODE_REJECT_REASON_SYSTEM_count', 'previous_application_CODE_REJECT_REASON_SYSTEM_count_norm', 'previous_application_CODE_REJECT_REASON_VERIF_count', 'previous_application_CODE_REJECT_REASON_VERIF_count_norm', 'previous_application_CODE_REJECT_REASON_XAP_count', 'previous_application_CODE_REJECT_REASON_XAP_count_norm', 'previous_application_CODE_REJECT_REASON_XNA_count', 'previous_application_CODE_REJECT_REASON_XNA_count_norm', 'previous_application_NAME_TYPE_SUITE_Children_count', 'previous_application_NAME_TYPE_SUITE_Children_count_norm', 'previous_application_NAME_TYPE_SUITE_Family_count', 'previous_application_NAME_TYPE_SUITE_Family_count_norm', 'previous_application_NAME_TYPE_SUITE_Group of people_count', 'previous_application_NAME_TYPE_SUITE_Group of people_count_norm', 'previous_application_NAME_TYPE_SUITE_Other_A_count', 'previous_application_NAME_TYPE_SUITE_Other_A_count_norm', 'previous_application_NAME_TYPE_SUITE_Other_B_count', 'previous_application_NAME_TYPE_SUITE_Other_B_count_norm', 'previous_application_NAME_TYPE_SUITE_Spouse, partner_count', 'previous_application_NAME_TYPE_SUITE_Spouse, partner_count_norm', 'previous_application_NAME_TYPE_SUITE_Unaccompanied_count', 'previous_application_NAME_TYPE_SUITE_Unaccompanied_count_norm', 'previous_application_NAME_CLIENT_TYPE_New_count', 'previous_application_NAME_CLIENT_TYPE_New_count_norm', 'previous_application_NAME_CLIENT_TYPE_Refreshed_count', 'previous_application_NAME_CLIENT_TYPE_Refreshed_count_norm', 'previous_application_NAME_CLIENT_TYPE_Repeater_count', 'previous_application_NAME_CLIENT_TYPE_Repeater_count_norm', 'previous_application_NAME_CLIENT_TYPE_XNA_count', 'previous_application_NAME_CLIENT_TYPE_XNA_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Additional Service_count', 'previous_application_NAME_GOODS_CATEGORY_Additional Service_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Animals_count', 'previous_application_NAME_GOODS_CATEGORY_Animals_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Audio/Video_count', 'previous_application_NAME_GOODS_CATEGORY_Audio/Video_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Auto Accessories_count', 'previous_application_NAME_GOODS_CATEGORY_Auto Accessories_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Clothing and Accessories_count', 'previous_application_NAME_GOODS_CATEGORY_Clothing and Accessories_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Computers_count', 'previous_application_NAME_GOODS_CATEGORY_Computers_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Construction Materials_count', 'previous_application_NAME_GOODS_CATEGORY_Construction Materials_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Consumer Electronics_count', 'previous_application_NAME_GOODS_CATEGORY_Consumer Electronics_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Direct Sales_count', 'previous_application_NAME_GOODS_CATEGORY_Direct Sales_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Education_count', 'previous_application_NAME_GOODS_CATEGORY_Education_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Fitness_count', 'previous_application_NAME_GOODS_CATEGORY_Fitness_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Furniture_count', 'previous_application_NAME_GOODS_CATEGORY_Furniture_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Gardening_count', 'previous_application_NAME_GOODS_CATEGORY_Gardening_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Homewares_count', 'previous_application_NAME_GOODS_CATEGORY_Homewares_count_norm', 'previous_application_NAME_GOODS_CATEGORY_House Construction_count', 'previous_application_NAME_GOODS_CATEGORY_House Construction_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Insurance_count', 'previous_application_NAME_GOODS_CATEGORY_Insurance_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Jewelry_count', 'previous_application_NAME_GOODS_CATEGORY_Jewelry_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Medical Supplies_count', 'previous_application_NAME_GOODS_CATEGORY_Medical Supplies_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Medicine_count', 'previous_application_NAME_GOODS_CATEGORY_Medicine_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Mobile_count', 'previous_application_NAME_GOODS_CATEGORY_Mobile_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Office Appliances_count', 'previous_application_NAME_GOODS_CATEGORY_Office Appliances_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Other_count', 'previous_application_NAME_GOODS_CATEGORY_Other_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_count', 'previous_application_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Sport and Leisure_count', 'previous_application_NAME_GOODS_CATEGORY_Sport and Leisure_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Tourism_count', 'previous_application_NAME_GOODS_CATEGORY_Tourism_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Vehicles_count', 'previous_application_NAME_GOODS_CATEGORY_Vehicles_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Weapon_count', 'previous_application_NAME_GOODS_CATEGORY_Weapon_count_norm', 'previous_application_NAME_GOODS_CATEGORY_XNA_count', 'previous_application_NAME_GOODS_CATEGORY_XNA_count_norm', 'previous_application_NAME_PORTFOLIO_Cards_count', 'previous_application_NAME_PORTFOLIO_Cards_count_norm', 'previous_application_NAME_PORTFOLIO_Cars_count', 'previous_application_NAME_PORTFOLIO_Cars_count_norm', 'previous_application_NAME_PORTFOLIO_Cash_count', 'previous_application_NAME_PORTFOLIO_Cash_count_norm', 'previous_application_NAME_PORTFOLIO_POS_count', 'previous_application_NAME_PORTFOLIO_POS_count_norm', 'previous_application_NAME_PORTFOLIO_XNA_count', 'previous_application_NAME_PORTFOLIO_XNA_count_norm', 'previous_application_NAME_PRODUCT_TYPE_XNA_count', 'previous_application_NAME_PRODUCT_TYPE_XNA_count_norm', 'previous_application_NAME_PRODUCT_TYPE_walk-in_count', 'previous_application_NAME_PRODUCT_TYPE_walk-in_count_norm', 'previous_application_NAME_PRODUCT_TYPE_x-sell_count', 'previous_application_NAME_PRODUCT_TYPE_x-sell_count_norm', 'previous_application_CHANNEL_TYPE_AP+ (Cash loan)_count', 'previous_application_CHANNEL_TYPE_AP+ (Cash loan)_count_norm', 'previous_application_CHANNEL_TYPE_Car dealer_count', 'previous_application_CHANNEL_TYPE_Car dealer_count_norm', 'previous_application_CHANNEL_TYPE_Channel of corporate sales_count', 'previous_application_CHANNEL_TYPE_Channel of corporate sales_count_norm', 'previous_application_CHANNEL_TYPE_Contact center_count', 'previous_application_CHANNEL_TYPE_Contact center_count_norm', 'previous_application_CHANNEL_TYPE_Country-wide_count', 'previous_application_CHANNEL_TYPE_Country-wide_count_norm', 'previous_application_CHANNEL_TYPE_Credit and cash offices_count', 'previous_application_CHANNEL_TYPE_Credit and cash offices_count_norm', 'previous_application_CHANNEL_TYPE_Regional / Local_count', 'previous_application_CHANNEL_TYPE_Regional / Local_count_norm', 'previous_application_CHANNEL_TYPE_Stone_count', 'previous_application_CHANNEL_TYPE_Stone_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Auto technology_count', 'previous_application_NAME_SELLER_INDUSTRY_Auto technology_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Clothing_count', 'previous_application_NAME_SELLER_INDUSTRY_Clothing_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Connectivity_count', 'previous_application_NAME_SELLER_INDUSTRY_Connectivity_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Construction_count', 'previous_application_NAME_SELLER_INDUSTRY_Construction_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Consumer electronics_count', 'previous_application_NAME_SELLER_INDUSTRY_Consumer electronics_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Furniture_count', 'previous_application_NAME_SELLER_INDUSTRY_Furniture_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Industry_count', 'previous_application_NAME_SELLER_INDUSTRY_Industry_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Jewelry_count', 'previous_application_NAME_SELLER_INDUSTRY_Jewelry_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_MLM partners_count', 'previous_application_NAME_SELLER_INDUSTRY_MLM partners_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Tourism_count', 'previous_application_NAME_SELLER_INDUSTRY_Tourism_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_XNA_count', 'previous_application_NAME_SELLER_INDUSTRY_XNA_count_norm', 'previous_application_NAME_YIELD_GROUP_XNA_count', 'previous_application_NAME_YIELD_GROUP_XNA_count_norm', 'previous_application_NAME_YIELD_GROUP_high_count', 'previous_application_NAME_YIELD_GROUP_high_count_norm', 'previous_application_NAME_YIELD_GROUP_low_action_count', 'previous_application_NAME_YIELD_GROUP_low_action_count_norm', 'previous_application_NAME_YIELD_GROUP_low_normal_count', 'previous_application_NAME_YIELD_GROUP_low_normal_count_norm', 'previous_application_NAME_YIELD_GROUP_middle_count', 'previous_application_NAME_YIELD_GROUP_middle_count_norm', 'previous_application_PRODUCT_COMBINATION_Card Street_count', 'previous_application_PRODUCT_COMBINATION_Card Street_count_norm', 'previous_application_PRODUCT_COMBINATION_Card X-Sell_count', 'previous_application_PRODUCT_COMBINATION_Card X-Sell_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash_count', 'previous_application_PRODUCT_COMBINATION_Cash_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash Street: high_count', 'previous_application_PRODUCT_COMBINATION_Cash Street: high_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash Street: low_count', 'previous_application_PRODUCT_COMBINATION_Cash Street: low_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash Street: middle_count', 'previous_application_PRODUCT_COMBINATION_Cash Street: middle_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: high_count', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: high_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: low_count', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: low_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: middle_count', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: middle_count_norm', 'previous_application_PRODUCT_COMBINATION_POS household with interest_count', 'previous_application_PRODUCT_COMBINATION_POS household with interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS household without interest_count', 'previous_application_PRODUCT_COMBINATION_POS household without interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS industry with interest_count', 'previous_application_PRODUCT_COMBINATION_POS industry with interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS industry without interest_count', 'previous_application_PRODUCT_COMBINATION_POS industry without interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS mobile with interest_count', 'previous_application_PRODUCT_COMBINATION_POS mobile with interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS mobile without interest_count', 'previous_application_PRODUCT_COMBINATION_POS mobile without interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS other with interest_count', 'previous_application_PRODUCT_COMBINATION_POS other with interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS others without interest_count', 'previous_application_PRODUCT_COMBINATION_POS others without interest_count_norm']
    previous_application_expected_columns_after_categorical_aggregation_with_curr = ['SK_ID_CURR', 'previous_application_NAME_CONTRACT_TYPE_Cash loans_count', 'previous_application_NAME_CONTRACT_TYPE_Cash loans_count_norm', 'previous_application_NAME_CONTRACT_TYPE_Consumer loans_count', 'previous_application_NAME_CONTRACT_TYPE_Consumer loans_count_norm', 'previous_application_NAME_CONTRACT_TYPE_Revolving loans_count', 'previous_application_NAME_CONTRACT_TYPE_Revolving loans_count_norm', 'previous_application_NAME_CONTRACT_TYPE_XNA_count', 'previous_application_NAME_CONTRACT_TYPE_XNA_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_FRIDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_FRIDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_MONDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_MONDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_SATURDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_SATURDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_SUNDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_SUNDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_THURSDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_THURSDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_TUESDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_TUESDAY_count_norm', 'previous_application_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_count', 'previous_application_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_count_norm', 'previous_application_FLAG_LAST_APPL_PER_CONTRACT_N_count', 'previous_application_FLAG_LAST_APPL_PER_CONTRACT_N_count_norm', 'previous_application_FLAG_LAST_APPL_PER_CONTRACT_Y_count', 'previous_application_FLAG_LAST_APPL_PER_CONTRACT_Y_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Business development_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Business development_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a garage_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a garage_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a home_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a home_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a new car_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a new car_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a used car_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Buying a used car_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Car repairs_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Car repairs_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Education_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Education_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Everyday expenses_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Everyday expenses_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Furniture_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Furniture_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Hobby_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Hobby_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Journey_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Journey_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Medicine_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Medicine_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Money for a third person_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Money for a third person_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Other_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Other_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Payments on other loans_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Payments on other loans_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Repairs_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Repairs_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Urgent needs_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Urgent needs_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_XAP_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_XAP_count_norm', 'previous_application_NAME_CASH_LOAN_PURPOSE_XNA_count', 'previous_application_NAME_CASH_LOAN_PURPOSE_XNA_count_norm', 'previous_application_NAME_CONTRACT_STATUS_Approved_count', 'previous_application_NAME_CONTRACT_STATUS_Approved_count_norm', 'previous_application_NAME_CONTRACT_STATUS_Canceled_count', 'previous_application_NAME_CONTRACT_STATUS_Canceled_count_norm', 'previous_application_NAME_CONTRACT_STATUS_Refused_count', 'previous_application_NAME_CONTRACT_STATUS_Refused_count_norm', 'previous_application_NAME_CONTRACT_STATUS_Unused offer_count', 'previous_application_NAME_CONTRACT_STATUS_Unused offer_count_norm', 'previous_application_NAME_PAYMENT_TYPE_Cash through the bank_count', 'previous_application_NAME_PAYMENT_TYPE_Cash through the bank_count_norm', 'previous_application_NAME_PAYMENT_TYPE_Cashless from the account of the employer_count', 'previous_application_NAME_PAYMENT_TYPE_Cashless from the account of the employer_count_norm', 'previous_application_NAME_PAYMENT_TYPE_Non-cash from your account_count', 'previous_application_NAME_PAYMENT_TYPE_Non-cash from your account_count_norm', 'previous_application_NAME_PAYMENT_TYPE_XNA_count', 'previous_application_NAME_PAYMENT_TYPE_XNA_count_norm', 'previous_application_CODE_REJECT_REASON_CLIENT_count', 'previous_application_CODE_REJECT_REASON_CLIENT_count_norm', 'previous_application_CODE_REJECT_REASON_HC_count', 'previous_application_CODE_REJECT_REASON_HC_count_norm', 'previous_application_CODE_REJECT_REASON_LIMIT_count', 'previous_application_CODE_REJECT_REASON_LIMIT_count_norm', 'previous_application_CODE_REJECT_REASON_SCO_count', 'previous_application_CODE_REJECT_REASON_SCO_count_norm', 'previous_application_CODE_REJECT_REASON_SCOFR_count', 'previous_application_CODE_REJECT_REASON_SCOFR_count_norm', 'previous_application_CODE_REJECT_REASON_SYSTEM_count', 'previous_application_CODE_REJECT_REASON_SYSTEM_count_norm', 'previous_application_CODE_REJECT_REASON_VERIF_count', 'previous_application_CODE_REJECT_REASON_VERIF_count_norm', 'previous_application_CODE_REJECT_REASON_XAP_count', 'previous_application_CODE_REJECT_REASON_XAP_count_norm', 'previous_application_CODE_REJECT_REASON_XNA_count', 'previous_application_CODE_REJECT_REASON_XNA_count_norm', 'previous_application_NAME_TYPE_SUITE_Children_count', 'previous_application_NAME_TYPE_SUITE_Children_count_norm', 'previous_application_NAME_TYPE_SUITE_Family_count', 'previous_application_NAME_TYPE_SUITE_Family_count_norm', 'previous_application_NAME_TYPE_SUITE_Group of people_count', 'previous_application_NAME_TYPE_SUITE_Group of people_count_norm', 'previous_application_NAME_TYPE_SUITE_Other_A_count', 'previous_application_NAME_TYPE_SUITE_Other_A_count_norm', 'previous_application_NAME_TYPE_SUITE_Other_B_count', 'previous_application_NAME_TYPE_SUITE_Other_B_count_norm', 'previous_application_NAME_TYPE_SUITE_Spouse, partner_count', 'previous_application_NAME_TYPE_SUITE_Spouse, partner_count_norm', 'previous_application_NAME_TYPE_SUITE_Unaccompanied_count', 'previous_application_NAME_TYPE_SUITE_Unaccompanied_count_norm', 'previous_application_NAME_CLIENT_TYPE_New_count', 'previous_application_NAME_CLIENT_TYPE_New_count_norm', 'previous_application_NAME_CLIENT_TYPE_Refreshed_count', 'previous_application_NAME_CLIENT_TYPE_Refreshed_count_norm', 'previous_application_NAME_CLIENT_TYPE_Repeater_count', 'previous_application_NAME_CLIENT_TYPE_Repeater_count_norm', 'previous_application_NAME_CLIENT_TYPE_XNA_count', 'previous_application_NAME_CLIENT_TYPE_XNA_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Additional Service_count', 'previous_application_NAME_GOODS_CATEGORY_Additional Service_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Animals_count', 'previous_application_NAME_GOODS_CATEGORY_Animals_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Audio/Video_count', 'previous_application_NAME_GOODS_CATEGORY_Audio/Video_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Auto Accessories_count', 'previous_application_NAME_GOODS_CATEGORY_Auto Accessories_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Clothing and Accessories_count', 'previous_application_NAME_GOODS_CATEGORY_Clothing and Accessories_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Computers_count', 'previous_application_NAME_GOODS_CATEGORY_Computers_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Construction Materials_count', 'previous_application_NAME_GOODS_CATEGORY_Construction Materials_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Consumer Electronics_count', 'previous_application_NAME_GOODS_CATEGORY_Consumer Electronics_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Direct Sales_count', 'previous_application_NAME_GOODS_CATEGORY_Direct Sales_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Education_count', 'previous_application_NAME_GOODS_CATEGORY_Education_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Fitness_count', 'previous_application_NAME_GOODS_CATEGORY_Fitness_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Furniture_count', 'previous_application_NAME_GOODS_CATEGORY_Furniture_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Gardening_count', 'previous_application_NAME_GOODS_CATEGORY_Gardening_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Homewares_count', 'previous_application_NAME_GOODS_CATEGORY_Homewares_count_norm', 'previous_application_NAME_GOODS_CATEGORY_House Construction_count', 'previous_application_NAME_GOODS_CATEGORY_House Construction_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Insurance_count', 'previous_application_NAME_GOODS_CATEGORY_Insurance_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Jewelry_count', 'previous_application_NAME_GOODS_CATEGORY_Jewelry_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Medical Supplies_count', 'previous_application_NAME_GOODS_CATEGORY_Medical Supplies_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Medicine_count', 'previous_application_NAME_GOODS_CATEGORY_Medicine_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Mobile_count', 'previous_application_NAME_GOODS_CATEGORY_Mobile_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Office Appliances_count', 'previous_application_NAME_GOODS_CATEGORY_Office Appliances_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Other_count', 'previous_application_NAME_GOODS_CATEGORY_Other_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_count', 'previous_application_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Sport and Leisure_count', 'previous_application_NAME_GOODS_CATEGORY_Sport and Leisure_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Tourism_count', 'previous_application_NAME_GOODS_CATEGORY_Tourism_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Vehicles_count', 'previous_application_NAME_GOODS_CATEGORY_Vehicles_count_norm', 'previous_application_NAME_GOODS_CATEGORY_Weapon_count', 'previous_application_NAME_GOODS_CATEGORY_Weapon_count_norm', 'previous_application_NAME_GOODS_CATEGORY_XNA_count', 'previous_application_NAME_GOODS_CATEGORY_XNA_count_norm', 'previous_application_NAME_PORTFOLIO_Cards_count', 'previous_application_NAME_PORTFOLIO_Cards_count_norm', 'previous_application_NAME_PORTFOLIO_Cars_count', 'previous_application_NAME_PORTFOLIO_Cars_count_norm', 'previous_application_NAME_PORTFOLIO_Cash_count', 'previous_application_NAME_PORTFOLIO_Cash_count_norm', 'previous_application_NAME_PORTFOLIO_POS_count', 'previous_application_NAME_PORTFOLIO_POS_count_norm', 'previous_application_NAME_PORTFOLIO_XNA_count', 'previous_application_NAME_PORTFOLIO_XNA_count_norm', 'previous_application_NAME_PRODUCT_TYPE_XNA_count', 'previous_application_NAME_PRODUCT_TYPE_XNA_count_norm', 'previous_application_NAME_PRODUCT_TYPE_walk-in_count', 'previous_application_NAME_PRODUCT_TYPE_walk-in_count_norm', 'previous_application_NAME_PRODUCT_TYPE_x-sell_count', 'previous_application_NAME_PRODUCT_TYPE_x-sell_count_norm', 'previous_application_CHANNEL_TYPE_AP+ (Cash loan)_count', 'previous_application_CHANNEL_TYPE_AP+ (Cash loan)_count_norm', 'previous_application_CHANNEL_TYPE_Car dealer_count', 'previous_application_CHANNEL_TYPE_Car dealer_count_norm', 'previous_application_CHANNEL_TYPE_Channel of corporate sales_count', 'previous_application_CHANNEL_TYPE_Channel of corporate sales_count_norm', 'previous_application_CHANNEL_TYPE_Contact center_count', 'previous_application_CHANNEL_TYPE_Contact center_count_norm', 'previous_application_CHANNEL_TYPE_Country-wide_count', 'previous_application_CHANNEL_TYPE_Country-wide_count_norm', 'previous_application_CHANNEL_TYPE_Credit and cash offices_count', 'previous_application_CHANNEL_TYPE_Credit and cash offices_count_norm', 'previous_application_CHANNEL_TYPE_Regional / Local_count', 'previous_application_CHANNEL_TYPE_Regional / Local_count_norm', 'previous_application_CHANNEL_TYPE_Stone_count', 'previous_application_CHANNEL_TYPE_Stone_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Auto technology_count', 'previous_application_NAME_SELLER_INDUSTRY_Auto technology_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Clothing_count', 'previous_application_NAME_SELLER_INDUSTRY_Clothing_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Connectivity_count', 'previous_application_NAME_SELLER_INDUSTRY_Connectivity_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Construction_count', 'previous_application_NAME_SELLER_INDUSTRY_Construction_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Consumer electronics_count', 'previous_application_NAME_SELLER_INDUSTRY_Consumer electronics_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Furniture_count', 'previous_application_NAME_SELLER_INDUSTRY_Furniture_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Industry_count', 'previous_application_NAME_SELLER_INDUSTRY_Industry_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Jewelry_count', 'previous_application_NAME_SELLER_INDUSTRY_Jewelry_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_MLM partners_count', 'previous_application_NAME_SELLER_INDUSTRY_MLM partners_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_Tourism_count', 'previous_application_NAME_SELLER_INDUSTRY_Tourism_count_norm', 'previous_application_NAME_SELLER_INDUSTRY_XNA_count', 'previous_application_NAME_SELLER_INDUSTRY_XNA_count_norm', 'previous_application_NAME_YIELD_GROUP_XNA_count', 'previous_application_NAME_YIELD_GROUP_XNA_count_norm', 'previous_application_NAME_YIELD_GROUP_high_count', 'previous_application_NAME_YIELD_GROUP_high_count_norm', 'previous_application_NAME_YIELD_GROUP_low_action_count', 'previous_application_NAME_YIELD_GROUP_low_action_count_norm', 'previous_application_NAME_YIELD_GROUP_low_normal_count', 'previous_application_NAME_YIELD_GROUP_low_normal_count_norm', 'previous_application_NAME_YIELD_GROUP_middle_count', 'previous_application_NAME_YIELD_GROUP_middle_count_norm', 'previous_application_PRODUCT_COMBINATION_Card Street_count', 'previous_application_PRODUCT_COMBINATION_Card Street_count_norm', 'previous_application_PRODUCT_COMBINATION_Card X-Sell_count', 'previous_application_PRODUCT_COMBINATION_Card X-Sell_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash_count', 'previous_application_PRODUCT_COMBINATION_Cash_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash Street: high_count', 'previous_application_PRODUCT_COMBINATION_Cash Street: high_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash Street: low_count', 'previous_application_PRODUCT_COMBINATION_Cash Street: low_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash Street: middle_count', 'previous_application_PRODUCT_COMBINATION_Cash Street: middle_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: high_count', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: high_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: low_count', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: low_count_norm', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: middle_count', 'previous_application_PRODUCT_COMBINATION_Cash X-Sell: middle_count_norm', 'previous_application_PRODUCT_COMBINATION_POS household with interest_count', 'previous_application_PRODUCT_COMBINATION_POS household with interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS household without interest_count', 'previous_application_PRODUCT_COMBINATION_POS household without interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS industry with interest_count', 'previous_application_PRODUCT_COMBINATION_POS industry with interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS industry without interest_count', 'previous_application_PRODUCT_COMBINATION_POS industry without interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS mobile with interest_count', 'previous_application_PRODUCT_COMBINATION_POS mobile with interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS mobile without interest_count', 'previous_application_PRODUCT_COMBINATION_POS mobile without interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS other with interest_count', 'previous_application_PRODUCT_COMBINATION_POS other with interest_count_norm', 'previous_application_PRODUCT_COMBINATION_POS others without interest_count', 'previous_application_PRODUCT_COMBINATION_POS others without interest_count_norm']


    previous_application_num_agg_SK_ID_CURR = agg_numeric(previous_application.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='previous_application', expected_columns_after_numeric_aggregation=previous_application_expected_columns_after_numeric_aggregation)
    previous_application_cat_agg_SK_ID_CURR = count_categorical(previous_application.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='previous_application', expected_columns_after_categorical_aggregation=previous_application_expected_columns_after_categorical_aggregation)

    # List of expected columns after aggregation in case input is empty for POS_CASH_balance

    POS_CASH_balance_expected_columns_after_numeric_aggregation = [
       'SK_ID_CURR', 'POS_CASH_balance_MONTHS_BALANCE_count',
       'POS_CASH_balance_MONTHS_BALANCE_mean',
       'POS_CASH_balance_MONTHS_BALANCE_max',
       'POS_CASH_balance_MONTHS_BALANCE_min',
       'POS_CASH_balance_MONTHS_BALANCE_sum',
       'POS_CASH_balance_CNT_INSTALMENT_count',
       'POS_CASH_balance_CNT_INSTALMENT_mean',
       'POS_CASH_balance_CNT_INSTALMENT_max',
       'POS_CASH_balance_CNT_INSTALMENT_min',
       'POS_CASH_balance_CNT_INSTALMENT_sum',
       'POS_CASH_balance_CNT_INSTALMENT_FUTURE_count',
       'POS_CASH_balance_CNT_INSTALMENT_FUTURE_mean',
       'POS_CASH_balance_CNT_INSTALMENT_FUTURE_max',
       'POS_CASH_balance_CNT_INSTALMENT_FUTURE_min',
       'POS_CASH_balance_CNT_INSTALMENT_FUTURE_sum',
       'POS_CASH_balance_SK_DPD_count', 'POS_CASH_balance_SK_DPD_mean',
       'POS_CASH_balance_SK_DPD_max', 'POS_CASH_balance_SK_DPD_min',
       'POS_CASH_balance_SK_DPD_sum', 'POS_CASH_balance_SK_DPD_DEF_count',
       'POS_CASH_balance_SK_DPD_DEF_mean', 'POS_CASH_balance_SK_DPD_DEF_max',
       'POS_CASH_balance_SK_DPD_DEF_min', 'POS_CASH_balance_SK_DPD_DEF_sum'
       ]
    
    POS_CASH_balance_expected_columns_after_categorical_aggregation = [
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Active_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Active_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Amortized debt_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Amortized debt_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Approved_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Approved_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Canceled_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Canceled_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Completed_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Completed_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Demand_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Demand_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Returned to the store_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Returned to the store_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Signed_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Signed_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_XNA_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_XNA_count_norm'
       ]
    
    POS_CASH_balance_expected_columns_after_categorical_aggregation_with_curr = [
       'SK_ID_CURR',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Active_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Active_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Amortized debt_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Amortized debt_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Approved_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Approved_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Canceled_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Canceled_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Completed_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Completed_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Demand_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Demand_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Returned to the store_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Returned to the store_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Signed_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_Signed_count_norm',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_XNA_count',
       'POS_CASH_balance_NAME_CONTRACT_STATUS_XNA_count_norm'
       ]

    POS_CASH_balance_num_agg_SK_ID_CURR = agg_numeric(POS_CASH_balance.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='POS_CASH_balance', expected_columns_after_numeric_aggregation=POS_CASH_balance_expected_columns_after_numeric_aggregation)
    POS_CASH_balance_cat_agg_SK_ID_CURR = count_categorical(POS_CASH_balance.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='POS_CASH_balance', expected_columns_after_categorical_aggregation=POS_CASH_balance_expected_columns_after_categorical_aggregation)

    # List of expected columns after aggregation in case input is empty for credit_card_balance

    credit_card_balance_expected_columns_after_numerical_aggregation = ['SK_ID_CURR', 'credit_card_balance_MONTHS_BALANCE_count', 'credit_card_balance_MONTHS_BALANCE_mean', 'credit_card_balance_MONTHS_BALANCE_max', 'credit_card_balance_MONTHS_BALANCE_min', 'credit_card_balance_MONTHS_BALANCE_sum', 'credit_card_balance_AMT_BALANCE_count', 'credit_card_balance_AMT_BALANCE_mean', 'credit_card_balance_AMT_BALANCE_max', 'credit_card_balance_AMT_BALANCE_min', 'credit_card_balance_AMT_BALANCE_sum', 'credit_card_balance_AMT_CREDIT_LIMIT_ACTUAL_count', 'credit_card_balance_AMT_CREDIT_LIMIT_ACTUAL_mean', 'credit_card_balance_AMT_CREDIT_LIMIT_ACTUAL_max', 'credit_card_balance_AMT_CREDIT_LIMIT_ACTUAL_min', 'credit_card_balance_AMT_CREDIT_LIMIT_ACTUAL_sum', 'credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_count', 'credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_mean', 'credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_max', 'credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_min', 'credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_sum', 'credit_card_balance_AMT_DRAWINGS_CURRENT_count', 'credit_card_balance_AMT_DRAWINGS_CURRENT_mean', 'credit_card_balance_AMT_DRAWINGS_CURRENT_max', 'credit_card_balance_AMT_DRAWINGS_CURRENT_min', 'credit_card_balance_AMT_DRAWINGS_CURRENT_sum', 'credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_count', 'credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_mean', 'credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_max', 'credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_min', 'credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_sum', 'credit_card_balance_AMT_DRAWINGS_POS_CURRENT_count', 'credit_card_balance_AMT_DRAWINGS_POS_CURRENT_mean', 'credit_card_balance_AMT_DRAWINGS_POS_CURRENT_max', 'credit_card_balance_AMT_DRAWINGS_POS_CURRENT_min', 'credit_card_balance_AMT_DRAWINGS_POS_CURRENT_sum', 'credit_card_balance_AMT_INST_MIN_REGULARITY_count', 'credit_card_balance_AMT_INST_MIN_REGULARITY_mean', 'credit_card_balance_AMT_INST_MIN_REGULARITY_max', 'credit_card_balance_AMT_INST_MIN_REGULARITY_min', 'credit_card_balance_AMT_INST_MIN_REGULARITY_sum', 'credit_card_balance_AMT_PAYMENT_CURRENT_count', 'credit_card_balance_AMT_PAYMENT_CURRENT_mean', 'credit_card_balance_AMT_PAYMENT_CURRENT_max', 'credit_card_balance_AMT_PAYMENT_CURRENT_min', 'credit_card_balance_AMT_PAYMENT_CURRENT_sum', 'credit_card_balance_AMT_PAYMENT_TOTAL_CURRENT_count', 'credit_card_balance_AMT_PAYMENT_TOTAL_CURRENT_mean', 'credit_card_balance_AMT_PAYMENT_TOTAL_CURRENT_max', 'credit_card_balance_AMT_PAYMENT_TOTAL_CURRENT_min', 'credit_card_balance_AMT_PAYMENT_TOTAL_CURRENT_sum', 'credit_card_balance_AMT_RECEIVABLE_PRINCIPAL_count', 'credit_card_balance_AMT_RECEIVABLE_PRINCIPAL_mean', 'credit_card_balance_AMT_RECEIVABLE_PRINCIPAL_max', 'credit_card_balance_AMT_RECEIVABLE_PRINCIPAL_min', 'credit_card_balance_AMT_RECEIVABLE_PRINCIPAL_sum', 'credit_card_balance_AMT_RECIVABLE_count', 'credit_card_balance_AMT_RECIVABLE_mean', 'credit_card_balance_AMT_RECIVABLE_max', 'credit_card_balance_AMT_RECIVABLE_min', 'credit_card_balance_AMT_RECIVABLE_sum', 'credit_card_balance_AMT_TOTAL_RECEIVABLE_count', 'credit_card_balance_AMT_TOTAL_RECEIVABLE_mean', 'credit_card_balance_AMT_TOTAL_RECEIVABLE_max', 'credit_card_balance_AMT_TOTAL_RECEIVABLE_min', 'credit_card_balance_AMT_TOTAL_RECEIVABLE_sum', 'credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_count', 'credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_mean', 'credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_max', 'credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_min', 'credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_sum', 'credit_card_balance_CNT_DRAWINGS_CURRENT_count', 'credit_card_balance_CNT_DRAWINGS_CURRENT_mean', 'credit_card_balance_CNT_DRAWINGS_CURRENT_max', 'credit_card_balance_CNT_DRAWINGS_CURRENT_min', 'credit_card_balance_CNT_DRAWINGS_CURRENT_sum', 'credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_count', 'credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_mean', 'credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_max', 'credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_min', 'credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_sum', 'credit_card_balance_CNT_DRAWINGS_POS_CURRENT_count', 'credit_card_balance_CNT_DRAWINGS_POS_CURRENT_mean', 'credit_card_balance_CNT_DRAWINGS_POS_CURRENT_max', 'credit_card_balance_CNT_DRAWINGS_POS_CURRENT_min', 'credit_card_balance_CNT_DRAWINGS_POS_CURRENT_sum', 'credit_card_balance_CNT_INSTALMENT_MATURE_CUM_count', 'credit_card_balance_CNT_INSTALMENT_MATURE_CUM_mean', 'credit_card_balance_CNT_INSTALMENT_MATURE_CUM_max', 'credit_card_balance_CNT_INSTALMENT_MATURE_CUM_min', 'credit_card_balance_CNT_INSTALMENT_MATURE_CUM_sum', 'credit_card_balance_SK_DPD_count', 'credit_card_balance_SK_DPD_mean', 'credit_card_balance_SK_DPD_max', 'credit_card_balance_SK_DPD_min', 'credit_card_balance_SK_DPD_sum', 'credit_card_balance_SK_DPD_DEF_count', 'credit_card_balance_SK_DPD_DEF_mean', 'credit_card_balance_SK_DPD_DEF_max', 'credit_card_balance_SK_DPD_DEF_min', 'credit_card_balance_SK_DPD_DEF_sum']

    credit_card_balance_expected_columns_after_categorical_aggregation = [
       'credit_card_balance_NAME_CONTRACT_STATUS_Active_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Active_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Approved_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Approved_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Completed_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Completed_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Demand_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Demand_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Refused_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Refused_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Sent proposal_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Sent proposal_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Signed_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Signed_count_norm'
       ]
    
    credit_card_balance_expected_columns_after_categorical_aggregation_with_curr = [
       'SK_ID_CURR',
       'credit_card_balance_NAME_CONTRACT_STATUS_Active_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Active_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Approved_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Approved_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Completed_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Completed_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Demand_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Demand_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Refused_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Refused_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Sent proposal_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Sent proposal_count_norm',
       'credit_card_balance_NAME_CONTRACT_STATUS_Signed_count',
       'credit_card_balance_NAME_CONTRACT_STATUS_Signed_count_norm'
       ]

    credit_card_balance_num_agg_SK_ID_CURR = agg_numeric(credit_card_balance.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='credit_card_balance', expected_columns_after_numeric_aggregation=credit_card_balance_expected_columns_after_numerical_aggregation)
    credit_card_balance_cat_agg_SK_ID_CURR = count_categorical(credit_card_balance.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='credit_card_balance', expected_columns_after_categorical_aggregation=credit_card_balance_expected_columns_after_categorical_aggregation)

    # List of expected columns after aggregation in case input is empty for installments_payments

    installments_payments_expected_columns_after_numerical_aggregation = [
       'SK_ID_CURR', 'credit_card_balance_NUM_INSTALMENT_VERSION_count',
       'credit_card_balance_NUM_INSTALMENT_VERSION_mean',
       'credit_card_balance_NUM_INSTALMENT_VERSION_max',
       'credit_card_balance_NUM_INSTALMENT_VERSION_min',
       'credit_card_balance_NUM_INSTALMENT_VERSION_sum',
       'credit_card_balance_NUM_INSTALMENT_NUMBER_count',
       'credit_card_balance_NUM_INSTALMENT_NUMBER_mean',
       'credit_card_balance_NUM_INSTALMENT_NUMBER_max',
       'credit_card_balance_NUM_INSTALMENT_NUMBER_min',
       'credit_card_balance_NUM_INSTALMENT_NUMBER_sum',
       'credit_card_balance_DAYS_INSTALMENT_count',
       'credit_card_balance_DAYS_INSTALMENT_mean',
       'credit_card_balance_DAYS_INSTALMENT_max',
       'credit_card_balance_DAYS_INSTALMENT_min',
       'credit_card_balance_DAYS_INSTALMENT_sum',
       'credit_card_balance_DAYS_ENTRY_PAYMENT_count',
       'credit_card_balance_DAYS_ENTRY_PAYMENT_mean',
       'credit_card_balance_DAYS_ENTRY_PAYMENT_max',
       'credit_card_balance_DAYS_ENTRY_PAYMENT_min',
       'credit_card_balance_DAYS_ENTRY_PAYMENT_sum',
       'credit_card_balance_AMT_INSTALMENT_count',
       'credit_card_balance_AMT_INSTALMENT_mean',
       'credit_card_balance_AMT_INSTALMENT_max',
       'credit_card_balance_AMT_INSTALMENT_min',
       'credit_card_balance_AMT_INSTALMENT_sum',
       'credit_card_balance_AMT_PAYMENT_count',
       'credit_card_balance_AMT_PAYMENT_mean',
       'credit_card_balance_AMT_PAYMENT_max',
       'credit_card_balance_AMT_PAYMENT_min',
       'credit_card_balance_AMT_PAYMENT_sum'
       ]
    

    installments_payments_num_agg_SK_ID_CURR = agg_numeric(installments_payments.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='credit_card_balance', expected_columns_after_numeric_aggregation=installments_payments_expected_columns_after_numerical_aggregation)

    dfs_to_merge = [
        previous_application_num_agg_SK_ID_CURR, previous_application_cat_agg_SK_ID_CURR,
        POS_CASH_balance_num_agg_SK_ID_CURR, POS_CASH_balance_cat_agg_SK_ID_CURR,
        credit_card_balance_num_agg_SK_ID_CURR, credit_card_balance_cat_agg_SK_ID_CURR,
        installments_payments_num_agg_SK_ID_CURR   
    ]

    list_of_expected_features_per_df = [
    previous_application_expected_columns_after_numeric_aggregation, previous_application_expected_columns_after_categorical_aggregation_with_curr, 
    POS_CASH_balance_expected_columns_after_numeric_aggregation, POS_CASH_balance_expected_columns_after_categorical_aggregation_with_curr, 
    credit_card_balance_expected_columns_after_numerical_aggregation, credit_card_balance_expected_columns_after_categorical_aggregation_with_curr,
    installments_payments_expected_columns_after_numerical_aggregation
    ]

    validated_dfs_to_merge = []

    for df_unchecked, expected_col_list in zip(dfs_to_merge, list_of_expected_features_per_df): #utiliser zip plutot que if and sinon ca va pas faire correspondre correctement les deux conditions par exemple ça va mettre df 2 avec liste de colonne 1
        if df_unchecked.empty :
            df_corrected = pd.DataFrame(columns=expected_col_list)
            validated_dfs_to_merge.append(df_corrected)
        else :
            validated_dfs_to_merge.append(df_unchecked)
    
    merge_key = 'SK_ID_CURR'

    merge_function = lambda left_df, right_df: pd.merge(left_df, right_df, on=merge_key, how='left')
    features_func_from_last_four = reduce(merge_function, validated_dfs_to_merge)
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

# --- CORRECTED AND SIMPLIFIED: Construct full model path directly ---
    pipeline_model_uri_to_use = None # Initialize
    try:
        # We know the tracking URI root is /app/mlruns (either from ENV or forced in code)
        # We know the experiment_id and run_id
        tracking_uri_root = "/app/mlruns" # This is consistently what it resolves to or is set to
        experiment_id = "448441841985485771" # From your directory structure
        # run_id_of_the_pipeline is already defined above as "dc6fb5c5f64e4a90b08ca76d9b962765"
        # pipeline_artifact_path is already defined above as "full_lgbm_pipeline"

        # The 'artifact_uri' from meta.yaml for your run is confirmed to be just 'artifacts'.
        # This means the artifacts for the run are in a subdirectory named 'artifacts'
        # relative to the run's main directory.

        # Path to the run's specific artifact folder:
        run_specific_artifacts_folder = os.path.join(
            tracking_uri_root,
            experiment_id,
            run_id_of_the_pipeline,
            "artifacts" # This comes from the fact that your meta.yaml's artifact_uri is 'artifacts'
        )

        # Path to the model directory itself (e.g., 'full_lgbm_pipeline') within that artifact folder:
        direct_model_path = os.path.join(
            run_specific_artifacts_folder,
            pipeline_artifact_path # This is 'full_lgbm_pipeline'
        )

        print(f"[DEBUG] Constructed direct model path for loading: {direct_model_path}")

        if not os.path.exists(direct_model_path):
            print(f"[CRITICAL_DEBUG] The constructed direct model path DOES NOT EXIST: {direct_model_path}")
            # For detailed debugging if the above fails, check parent directories:
            if os.path.exists(run_specific_artifacts_folder):
                 print(f"[CRITICAL_DEBUG] Parent artifact dir '{run_specific_artifacts_folder}' EXISTS. Contents: {os.listdir(run_specific_artifacts_folder)}")
            elif os.path.exists(os.path.dirname(run_specific_artifacts_folder)): # Run directory
                 print(f"[CRITICAL_DEBUG] Run directory '{os.path.dirname(run_specific_artifacts_folder)}' EXISTS. Contents: {os.listdir(os.path.dirname(run_specific_artifacts_folder))}")
            elif os.path.exists(os.path.dirname(os.path.dirname(run_specific_artifacts_folder))): # Experiment directory
                 print(f"[CRITICAL_DEBUG] Experiment directory '{os.path.dirname(os.path.dirname(run_specific_artifacts_folder))}' EXISTS. Contents: {os.listdir(os.path.dirname(os.path.dirname(run_specific_artifacts_folder)))}")
            else:
                print(f"[CRITICAL_DEBUG] Even the base '/app/mlruns' or experiment directory might not exist as expected.")
            raise FileNotFoundError(f"Manually constructed model path not found: {direct_model_path}")
        
        # Also check for MLmodel file specifically
        mlmodel_file_in_direct_path = os.path.join(direct_model_path, "MLmodel")
        if not os.path.exists(mlmodel_file_in_direct_path):
            print(f"[CRITICAL_DEBUG] MLmodel file DOES NOT EXIST at: {mlmodel_file_in_direct_path}")
            print(f"[CRITICAL_DEBUG] Contents of '{direct_model_path}': {os.listdir(direct_model_path) if os.path.exists(direct_model_path) else 'Path does not exist'}")
            raise FileNotFoundError(f"MLmodel file not found in constructed model path: {mlmodel_file_in_direct_path}")
        else:
            print(f"[DEBUG] MLmodel file confirmed to exist at: {mlmodel_file_in_direct_path}")


        pipeline_model_uri_to_use = direct_model_path # This is the path to the model's directory
        print(f"[DEBUG] Using direct path for model loading: {pipeline_model_uri_to_use}")

    except Exception as e_construct_path: # Changed variable name for clarity
        print(f"[DEBUG] ERROR trying to construct direct model path: {e_construct_path}")
        traceback.print_exc()
        raise # Re-raise to stop execution if path construction fails
    # --- END CORRECTED AND SIMPLIFIED ---

    # The old MLflow Path Debugging block (paths_to_check loop) can be removed if this new block works,
    # as this new block is more targeted and uses the same logic.
    # For now, I'll leave your existing path debugging block after this new one,
    # but it might be redundant if the above "CRITICAL_DEBUG" provide enough info.

    # This debug print should show file:///app/mlruns due to the forcing logic above it.
    # print(f"[PREDICT_SCORES_DEBUG] Current MLflow tracking URI: {mlflow.get_tracking_uri()}")
    # The lines above this comment regarding `current_tracking_uri` and `expected_tracking_uri_in_container`
    # and the `paths_to_check` loop might be redundant now or can be removed if the new block above works.
    # Let's comment out the old extensive path checking for now to reduce log noise.
    # experiment_id_for_path = "448441841985485771"
    # base_dir_to_check = "/app"
    # print("[DEBUG] --- Verifying Expected File System Structure Inside Container (OLD BLOCK - CAN BE REMOVED) ---")
    # ... (paths_to_check loop was here) ...
    # print("[DEBUG] --- End of File System Structure Verification (OLD BLOCK) ---")


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