import os
from src.config import MODEL_NAMES

# Project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
paths = {}

# Data directories
INPUT_PATH     = os.path.join(project_root, 'data', 'conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv')
train_data_dir = os.path.join(project_root, 'data', 'train')
val_data_dir   = os.path.join(project_root, 'data', 'val')

# Paths for fold files
X_train_paths = [os.path.join(train_data_dir, f'X_train_fold{fold+1}.csv') for fold in range(5)]
y_train_paths = [os.path.join(train_data_dir, f'y_train_fold{fold+1}.csv') for fold in range(5)]
X_val_paths   = [os.path.join(val_data_dir  , f'X_val_fold{fold+1}.csv')   for fold in range(5)]
y_val_paths   = [os.path.join(val_data_dir  , f'y_val_fold{fold+1}.csv')   for fold in range(5)]

paths.update({
    'X_train_paths': X_train_paths,
    'y_train_paths': y_train_paths,
    'X_val_paths'  : X_val_paths,
    'y_val_paths'  : y_val_paths,
})

# Model paths
for model_name in MODEL_NAMES:
    model_dir = os.path.join(project_root, 'results', model_name)
    model_paths = {
        'best_params': os.path.join(model_dir, 'best_params.json'),
        'pred_5folds': os.path.join(model_dir, 'predictions_5folds.csv'),
        'shap_fi'    : os.path.join(model_dir, 'shap_feature_importance.csv'),
    }

    # Catboost model file ending is .cbm and this classifier has feature importance
    if model_name == 'catboost':
        model_paths.update({
            'models': [os.path.join(model_dir, 'models', f'fold{fold+1}.cbm') for fold in range(5)],
            'feature_importance': os.path.join(model_dir, 'feature_importance.csv'),
        })
    # XGBoost model file ending is .json and this classifier has feature importance
    elif model_name == 'xgboost':
        model_paths.update({
            'models': [os.path.join(model_dir, 'models', f'fold{fold+1}.json') for fold in range(5)],
            'feature_importance': os.path.join(model_dir, 'feature_importance.csv'),
        })
    # LGBM and RF model file endings are .joblib and these classifiers have feature importance
    elif model_name in ['lgbm', 'rf']:
        model_paths.update({
            'models': [os.path.join(model_dir, 'models', f'fold{fold+1}.joblib') for fold in range(5)],
            'feature_importance': os.path.join(model_dir, 'feature_importance.csv'),
        })
    # SVM and LR model file ending are .joblib and these classifiers do not have feature importance
    else:
        model_paths.update({
            'models': [os.path.join(model_dir, 'models', f'fold{fold+1}.joblib') for fold in range(5)],
        })

    paths[model_name] = model_paths


if __name__ == '__main__':
    import json

    with open(os.path.join(project_root, 'paths.json'), 'w') as f:
        json.dump(paths, f, indent=4)