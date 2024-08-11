from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix
)

def calculate_metrics(y_true, y_pred, y_pred_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'f1_score': f1_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'specificity': specificity,
    }