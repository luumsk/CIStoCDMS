import os

# Variables
TARGET       = 'group'
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Paths
INPUT_PATH    = 'data/conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv'
X_train_paths = [f'data/train/X_train_fold{fold+1}.csv' for fold in range(5)]
y_train_paths = [f'data/train/y_train_fold{fold+1}.csv' for fold in range(5)]
X_val_paths   = [f'data/val/X_val_fold{fold+1}.csv' for fold in range(5)]
y_val_paths   = [f'data/val/y_val_fold{fold+1}.csv' for fold in range(5)]

paths = {
    'catboost': {
        'best_params': 'data/catboost/best_params.json',
        'pred_5folds': 'data/catboost/predictions_5folds.csv',
        'models': [f'data/catboost/models/fold{fold+1}.cbm' for fold in range(5)],
        'shap_fi': 'data/catboost/shap_feature_importance.csv',
        'feature_importance': 'data/catboost/feature_importance.csv'
    },
    'xgboost': {
        'best_params': 'data/xgboost/best_params.json',
        'pred_5folds': 'data/xgboost/predictions_5folds.csv',
        'models': [f'data/xgboost/models/fold{fold+1}.json' for fold in range(5)],
        'shap_fi': 'data/xgboost/shap_feature_importance.csv',
        'feature_importance': 'data/xgboost/feature_importance.csv'
    },
    'lgbm': {
        'best_params': 'data/lgbm/best_params.json',
        'pred_5folds': 'data/lgbm/predictions_5folds.csv',
        'models': [f'data/lgbm/models/fold{fold+1}.joblib' for fold in range(5)],
        'shap_fi': 'data/lgbm/shap_feature_importance.csv',
        'feature_importance': 'data/lgbm/feature_importance.csv'
    },
    'rf': {
        'best_params': 'data/rf/best_params.json',
        'pred_5folds': 'data/rf/predictions_5folds.csv',
        'models': [f'data/rf/models/fold{fold+1}.joblib' for fold in range(5)],
        'shap_fi': 'data/rf/shap_feature_importance.csv',
    },
    'svm': {
        'best_params': 'data/svm/best_params.json',
        'pred_5folds': 'data/svm/predictions_5folds.csv',
        'models': [f'data/svm/models/fold{fold+1}.joblib' for fold in range(5)],
        'shap_fi': 'data/svm/shap_feature_importance.csv',
    },
    'lr': {
        'best_params': 'data/lr/best_params.json',
        'pred_5folds': 'data/lr/predictions_5folds.csv',
        'models': [f'data/lr/models/fold{fold+1}.joblib' for fold in range(5)],
        'shap_fi': 'data/lr/shap_feature_importance.csv',
    }
}
