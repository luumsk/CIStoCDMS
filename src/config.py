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
        'best_params': 'results/catboost/best_params.json',
        'pred_5folds': 'results/catboost/predictions_5folds.csv',
        'models': [f'results/catboost/models/fold{fold+1}.cbm' for fold in range(5)],
        'shap_fi': 'results/catboost/shap_feature_importance.csv',
        'feature_importance': 'results/catboost/feature_importance.csv'
    },
    'xgboost': {
        'best_params': 'results/xgboost/best_params.json',
        'pred_5folds': 'results/xgboost/predictions_5folds.csv',
        'models': [f'results/xgboost/models/fold{fold+1}.json' for fold in range(5)],
        'shap_fi': 'results/xgboost/shap_feature_importance.csv',
        'feature_importance': 'results/xgboost/feature_importance.csv'
    },
    'lgbm': {
        'best_params': 'results/lgbm/best_params.json',
        'pred_5folds': 'results/lgbm/predictions_5folds.csv',
        'models': [f'results/lgbm/models/fold{fold+1}.joblib' for fold in range(5)],
        'shap_fi': 'results/lgbm/shap_feature_importance.csv',
        'feature_importance': 'results/lgbm/feature_importance.csv'
    },
    'rf': {
        'best_params': 'results/rf/best_params.json',
        'pred_5folds': 'results/rf/predictions_5folds.csv',
        'models': [f'results/rf/models/fold{fold+1}.joblib' for fold in range(5)],
        'shap_fi': 'results/rf/shap_feature_importance.csv',
        'feature_importance': 'results/rf/feature_importance.csv'
    },
    'svm': {
        'best_params': 'data/svm/best_params.json',
        'pred_5folds': 'data/svm/predictions_5folds.csv',
        'models': [f'data/svm/models/fold{fold+1}.joblib' for fold in range(5)],
        'shap_fi': 'data/svm/shap_feature_importance.csv',
    },
    'lr': {
        'best_params': 'results/lr/best_params.json',
        'pred_5folds': 'results/lr/predictions_5folds.csv',
        'models': [f'results/lr/models/fold{fold+1}.joblib' for fold in range(5)],
        'shap_fi': 'results/lr/shap_feature_importance.csv',
    }
}
