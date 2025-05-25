# tests/test_preprocessing_functions.py
import pytest # Pas toujours nécessaire pour des tests simples avec assert, mais bonne pratique
import pandas as pd
import numpy as np

# Importez les fonctions que vous voulez tester depuis votre module
# Pour que cela fonctionne, assurez-vous que your_project_folder est dans PYTHONPATH
# ou que vous exécutez pytest depuis your_project_folder
from preprocessing_pipeline import sanitize_lgbm_colname, agg_numeric # Ajoutez count_categorical, etc.

# --- Tests pour sanitize_lgbm_colname ---
def test_sanitize_simple_string():
    assert sanitize_lgbm_colname("Simple Name") == "Simple_Name"

def test_sanitize_with_special_chars():
    assert sanitize_lgbm_colname("col[1]: 'Test'") == "col_1_Test"

def test_sanitize_leading_number():
    assert sanitize_lgbm_colname("123_Test") == "_123_Test"

def test_sanitize_empty_string():
    # Le hash sera différent à chaque fois, donc on vérifie le pattern
    assert sanitize_lgbm_colname("").startswith("col_")

def test_sanitize_only_special_chars():
    assert sanitize_lgbm_colname("[]{}").startswith("col_") # ou devient '_' selon la logique exacte

# --- Tests pour agg_numeric (exemple) ---
# Vous aurez besoin de créer des DataFrames d'exemple
def test_agg_numeric_basic():
    data = {
        'SK_ID_GROUP': [1, 1, 2, 2, 3],
        'Numeric_Col1': [10, 20, 30, 40, 50],
        'Numeric_Col2': [1, 2, 3, 4, 5],
        'SK_ID_OTHER': [101, 102, 103, 104, 105] # Sera droppée par la fonction
    }
    df = pd.DataFrame(data)
    
    agg_df = agg_numeric(df.copy(), group_var='SK_ID_GROUP', df_name='test_df')
    
    # Vérifiez la forme du DataFrame agrégé
    assert agg_df.shape[0] == 3 # 3 groupes uniques
    
    # Vérifiez que les colonnes attendues sont créées (les noms exacts dépendent de votre fonction)
    # Exemple : 'test_df_Numeric_Col1_mean', 'test_df_Numeric_Col1_sum', etc.
    expected_cols_suffixes = ['_count', '_mean', '_max', '_min', '_sum']
    for col_base in ['Numeric_Col1', 'Numeric_Col2']:
        for suffix in expected_cols_suffixes:
            assert f'test_df_{col_base}{suffix}' in agg_df.columns
    
    # Vérifiez quelques valeurs spécifiques
    # Pour SK_ID_GROUP == 1, moyenne de Numeric_Col1 devrait être 15
    assert agg_df[agg_df['SK_ID_GROUP'] == 1]['test_df_Numeric_Col1_mean'].iloc[0] == 15.0
    # Pour SK_ID_GROUP == 2, somme de Numeric_Col2 devrait être 7
    assert agg_df[agg_df['SK_ID_GROUP'] == 2]['test_df_Numeric_Col2_sum'].iloc[0] == 7.0

# Vous ajouterez des tests similaires pour count_categorical et des parties de prepare_input_data

# Exemple de test pour une partie de prepare_input_data (plus complexe)
# Vous pourriez tester la logique de one-hot encoding sur current_app par exemple.
# Ou la logique de création d'une feature spécifique.
# Cela demande de créer des mini-DataFrames d'entrée pour current_app, bureau, etc.