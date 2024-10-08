import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from .config import *
from .paths import paths
from .helpers import load_joblib


np.random.seed(RANDOM_STATE)

def get_df(path=None, edss=False, mri=False):
    path = paths.get('input_path') if path is None else path
    df = pd.read_csv(path)

    cols_to_drop = []
    if 'Unnamed: 0' in df.columns:
        cols_to_drop.append('Unnamed: 0')
    if not edss:
        cols_to_drop += ['Initial_EDSS', 'Final_EDSS']
    if not mri:
        cols_to_drop += [col for col in df.columns if 'MRI' in col.upper()]

    df.drop(columns=cols_to_drop, inplace=True)

    # Convert target column to binary (0 and 1)
    # Original: 1 is MS, 2 is no MS
    # New: 1 is MS, 0 is no MS
    df[TARGET] = df[TARGET].replace({2.0: 0.0})
    
    return df

def get_Xy(df):
    y = df[TARGET]
    X = df.drop(columns=TARGET)
    return X, y

def split(X, y, n_folds=5):
    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val  , y_val   = X.iloc[val_idx]  , y.iloc[val_idx]

        # Save fold data using predefined paths
        X_train.to_csv(paths.get('X_train_paths')[fold], index=False)
        y_train.to_csv(paths.get('y_train_paths')[fold], index=False)
        X_val.to_csv(paths.get('X_val_paths')[fold], index=False)
        y_val.to_csv(paths.get('y_val_paths')[fold], index=False)

def load_data_fold(fold):
    X_train = pd.read_csv(paths.get('X_train_paths')[fold])
    y_train = pd.read_csv(paths.get('y_train_paths')[fold])
    X_val   = pd.read_csv(paths.get('X_val_paths')[fold])
    y_val   = pd.read_csv(paths.get('y_val_paths')[fold])
    return X_train, y_train, X_val, y_val

def load_model_fold(fold, model_name):
    if model_name not in MODEL_NAMES:
        raise ValueError("Incorrect model_name. Only support one of: ['catboost', 'xgboost', 'lgbm', 'rf', 'svm', 'lr']")

    model_paths = paths.get(model_name, {}).get('models')    

    if not model_paths:
        raise ValueError(f'Model paths not found for model {model_name}')

    model_path = model_paths[fold]
    model = None

    if model_name == 'catboost':
        model = CatBoostClassifier()
        model.load_model(model_path)
    elif model_name == 'xgboost':
        model = XGBClassifier()
        model.load_model(model_path)
    elif model_name in ['rf', 'lgbm', 'svm', 'lr']:
        model = load_joblib(model_path)

    return model

        
class BasicPreprocessor:
    def __init__(self):
        # Mapping of symptoms
        self.sym_map = {
            #    V  S  M  O
            1 : [1, 0, 0, 0], # Visual
            2 : [0, 1, 0, 0], # Sensory
            3 : [0, 0, 1, 0], # Motor
            4 : [0, 0, 0, 1], # Other
            5 : [1, 1, 0, 0], # Visual and Sensory
            6 : [1, 0, 1, 0], # Visual and Motor
            7 : [1, 0, 0, 1], # Visual and Other
            8 : [0, 1, 1, 0], # Sensory and Motor
            9 : [0, 1, 0, 1], # Sensory and Other
            10: [0, 0, 1, 1], # Motor and Other
            11: [1, 1, 1, 0], # Visual, Sensory and Motor
            12: [1, 1, 0, 1], # Visual, Sensory and Other
            13: [1, 0, 1, 1], # Visual, Motor and Other
            14: [0, 1, 1, 1], # Sensory, Motor and Other
            15: [1, 1, 1, 1], # Visual, Sensory, Motor and Other
        }

        # Mapping of monosymtomatic or polysymtomatic
        self.mp_map = {
            #   M  P
            1: [1, 0], # Monosymptomatic
            2: [0, 1], # Polysymptomatic
            3: [0, 0], # Unknown
        }


    def impute(self, df_, columns, method='mean'):
        df = df_.copy()
        for col in columns:
            if method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif method == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif method == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
        return df
    
    def map_initial_symptom(self, row):
        symptom = row['Initial_Symptom']
        bin_arr = self.sym_map.get(symptom)
        row['Symptom_Vision']  = bin_arr[0]
        row['Symptom_Sensory'] = bin_arr[1]
        row['Symptom_Motor']   = bin_arr[2]
        row['Symptom_Other']   = bin_arr[3]
        return row
    
    def encode_inital_symptom(self, df_):
        df = df_.copy()

        for new_col in ['Symptom_Vision', 'Symptom_Sensory', 'Symptom_Motor', 'Symptom_Other']:
            df[new_col] = np.nan
        df = df.apply(self.map_initial_symptom, axis=1)
        df.drop(columns='Initial_Symptom', inplace=True)

        return df

    def map_mono_poly(self, row):
        val = row['Mono_or_Polysymptomatic']
        bin_arr = self.mp_map.get(val)
        row['Mono_Symptomatic'] = bin_arr[0]
        row['Poly_Symptomatic'] = bin_arr[1]
        return row

    def encode_mono_poly(self, df_):
        df = df_.copy()

        for new_col in ['Mono_Symptomatic', 'Poly_Symptomatic']:
            df[new_col] = np.nan
        df = df.apply(self.map_mono_poly, axis=1)
        df.drop(columns='Mono_or_Polysymptomatic', inplace=True)

        return df
    
    def convert_to_binary(self, x):
        if x == 2.0: return 0.0  # No/Negative
        if x == 3.0: return -1.0 # Unknown
        return x

    def to_binary(self, df, columns):
        print(f'Reset values of these columns to binary: {columns}')
        for col in columns:
            df[col] = df[col].apply(self.convert_to_binary)
        return df

    def is_binary(self, arr):
        for x in arr:
            if x < 0.0 or x > 1.0:
                return False
        return True

    def find_cols_not_binary(self, df):
        cols = []
        exceptions = [TARGET, 'Age', 'Schooling', 'Oligoclonal_Bands']
        cols_to_check = [col for col in df.columns if col not in exceptions]

        print("Columns that contain numerical non-binary values:")
        for col in cols_to_check:
            unique_vals = df[col].unique()
            if not self.is_binary(unique_vals):
                cols.append(col)
                print('\t', col, unique_vals)
        return cols
    
    def scale(self, df_, columns, scaler_name='MinMaxScaler'):
        scaler = None

        if scaler_name == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler_name == 'StandardScaler':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Scaler '{scaler_name}' is unsupported.")

        df = df_.copy()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def __call__(self, df):
        # Print columns that have missing values
        missing_value_columns = df.isna().sum()[df.isna().sum() > 0].index.tolist()
        print(f'Missing value columns: {missing_value_columns}')

        # Impute missing values
        df = self.impute(df, columns=['Initial_Symptom'], method='mode') # categorical data
        df = self.impute(df, columns=['Schooling'], method='median') # numerical data

        # Split some numerical columns to a combination of binary columns
        df = self.encode_inital_symptom(df)
        df = self.encode_mono_poly(df)

        # Convert unknown values of 'Oligoclonal_Bands' from 2.0 to -1.0
        df['Oligoclonal_Bands'] = df['Oligoclonal_Bands'].map(lambda x: -1.0 if x == 2.0 else x)

        # Convert other numerical columns to binary (Unknown value is mapped to -1.0)
        df = self.to_binary(df, columns=self.find_cols_not_binary(df))

        # Scale values to same range
        # df = self.scale(df, columns=['Schooling', 'Age'])

        # Convert to dtype float
        df = df.astype('float64')
        
        return df
