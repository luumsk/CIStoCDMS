import pandas as pd
from .helpers import get_mean_by_key, save_joblib
from .metrics import calculate_metrics


def average_feature_importances(feature_importance_list):
    concatenated_df = pd.concat(feature_importance_list)
    mean_importances_df = concatenated_df.groupby('Feature Id').mean()
    return mean_importances_df.sort_values(by='Importances', ascending=False).reset_index()


class BaseTrainer:
    def __init__(self, model):
        self.model = model
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.y_pred = None
        self.y_pred_proba = None

    def check_training_data(self):
        if self.X_train is None or self.y_train is None:
            print('WARNING: X_train and y_train are None. Cannot train model.')
            return False
        return True
    
    def check_validation_data(self):
        if self.X_val is None or self.y_val is None:
            print('WARNING: X_val and y_val are None. Cannot evaluate model.')
            return False
        return True

    def set_input_data(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def set_output_data(self, y_pred, y_pred_proba):
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

    def save_model(self, model_path):
        save_joblib(path=model_path, data=self.model)

    def train(self):
        if self.check_training_data():
            self.model.fit(self.X_train, self.y_train)

    def predict(self):
        if self.X_val is not None:
            y_pred = self.model.predict(self.X_val)
            y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
            return y_pred, y_pred_proba
        return None, None
        
    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def perform_fold(self,
                     X_train, y_train, 
                     X_val, y_val, 
                     model_path=None, verbose=0):
        self.set_input_data(X_train, y_train, X_val, y_val)
        self.train()

        if model_path:
            self.save_model(model_path)
            print(f'Saved model to {model_path}')

        y_pred, y_pred_proba = self.predict()
        self.set_output_data(y_pred, y_pred_proba)
        
        metrics = calculate_metrics(self.y_val, self.y_pred, self.y_pred_proba)
        if verbose:
            self._print_metrics(metrics)

        feature_importance = self.get_feature_importance()
        return metrics, y_pred, y_pred_proba, feature_importance

    def perform_cross_validation(self, 
                                 X_train_list, y_train_list, 
                                 X_val_list, y_val_list, 
                                 model_path_list=None, 
                                 verbose=0):
        n_folds = len(X_train_list)

        self._validate_fold_lengths(X_train_list, y_train_list, 
                                    X_val_list, y_val_list, n_folds)

        if model_path_list is None:
            model_path_list = [None] * n_folds

        metrics_list, y_pred_list, y_pred_proba_list, feature_importance_list = [], [], [], []

        for fold in range(n_folds):
            if verbose: 
                print(f'Fold {fold + 1}/{n_folds}:')

            metrics, y_pred, y_pred_proba, feature_importance = self.perform_fold(
                X_train_list[fold],
                y_train_list[fold],
                X_val_list[fold],
                y_val_list[fold],
                model_path_list[fold],
                verbose
            )

            metrics_list.append(metrics)
            y_pred_list.append(y_pred)
            y_pred_proba_list.append(y_pred_proba)
            feature_importance_list.append(feature_importance)
            
        mean_metrics = get_mean_by_key(metrics_list)
        if verbose:
            self._print_mean_metrics(mean_metrics)

        return mean_metrics, y_pred_list, y_pred_proba_list, feature_importance_list

    @staticmethod
    def _validate_fold_lengths(X_train_list, y_train_list, 
                               X_val_list, y_val_list, n_folds):
        assert len(X_train_list) == n_folds, 'Mismatch in number of folds for X_train.'
        assert len(y_train_list) == n_folds, 'Mismatch in number of folds for y_train.'
        assert len(X_val_list)   == n_folds, 'Mismatch in number of folds for X_val.'
        assert len(y_val_list)   == n_folds, 'Mismatch in number of folds for y_val.'

    @staticmethod
    def _print_metrics(metrics):
        for key, value in metrics.items():
            print(f'\t- {key.upper()}: {value:.4f}')
        print('')

    @staticmethod
    def _print_mean_metrics(mean_metrics):
        print('-' * 30)
        for key, value in mean_metrics.items():
            print(f'\t- {key.upper()}: {value:.4f}')


class CatBoostTrainer(BaseTrainer):
    def save_model(self, model_path):
        self.model.save_model(model_path)

    def train(self):
        if self.check_training_data():
            if self.check_validation_data():
                self.model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    use_best_model=True,
                    verbose=0
                )
            else:
                self.model.fit(self.X_train, self.y_train, verbose=0)


class LGBMTrainer(BaseTrainer):  
    def train(self):
        if self.check_training_data():
            if self.check_validation_data():
                self.model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                )
            else:
                self.model.fit(self.X_train, self.y_train)
