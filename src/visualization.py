import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from .config import RANDOM_STATE

np.random.seed(RANDOM_STATE)

def get_catboost_feature_importance(model):
    feature_importance = model.get_feature_importance(prettified=True)
    return feature_importance  


def plot_calibration_curve(classifier, X, y, n_bins=10, fold=None):
    # Fit the classifier
    classifier.fit(X, y)

    # Predict probabilities
    y_pred_proba = classifier.predict_proba(X)[:, 1]

    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y, y_pred_proba, n_bins=n_bins)

    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')  # Perfectly calibrated line
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    if fold:
        plt.title(f'Calibration Curve - Fold {fold}')
    else:
        plt.title('Calibration Curve')
    plt.grid(True)
    plt.show()


def plot_pair_hist(df1, df2, columns, bins=10, titles=None):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    if titles is None:
        titles = ['Original Data', 'Augmented Data']
    
    df1.hist(column=columns, ax=ax1, bins=bins)
    ax1.set_title(titles[0])
    ax1.grid(False)
    ax1.set_yticklabels([])
    ax1.tick_params(axis='y', left=False, labelleft=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    df2.hist(column=columns, ax=ax2, bins=bins, color='orange')
    ax2.set_title(titles[1])
    ax2.grid(False)
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', left=False, labelleft=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_pair_pie(df1, df2, columns, titles=None):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    if titles is None:
        titles = ['Original Data', 'Augmented Data']

    df1[columns].value_counts().plot(
        kind='pie',
        ax=ax1,
        startangle=90,
        autopct='%1.1f%%',
        labels=['No CDMS', 'CDMS']
    )
    ax1.set_title(titles[0])
    ax1.set_ylabel('')

    df2[columns].value_counts().plot(
        kind='pie',
        ax=ax2,
        startangle=90,
        autopct='%1.1f%%',
        labels=['No CDMS', 'CDMS']
    )
    ax2.set_title(titles[1])
    ax2.set_ylabel('')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_folds(y_test_list, y_pred_list):
    _, axes = plt.subplots(3, 2, figsize=(10, 15), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten the 2D array of axes into 1D for easier iteration
    i = 0

    for y_test, y_preds in zip(y_test_list, y_pred_list):
        cm = confusion_matrix(y_test, y_preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            square=True, cbar=False,
            xticklabels=["Predicted Non-CDMS", "Predicted CDMS"],
            yticklabels=["Actual Non-CDMS", "Actual CDMS"],
            ax=axes[i]
        )
        axes[i].set_title(f'Fold {i+1}')
        i += 1

    # Hide the last axis
    axes[-1].axis('off')


def plot_catboost_feature_importance_folds(fi_list, columns_folds):
    fig, axes = plt.subplots(3, 2, figsize=(25, 15))
    axes = axes.flatten()  # Flatten the 2D array of axes into 1D for easier iteration
    i = 0

    for feature_importance, columns in zip(fi_list, columns_folds):
        sorted_idx = np.argsort(feature_importance)

        axes[i].barh(
            range(len(sorted_idx)),
            feature_importance[sorted_idx],
            align='center'
        )
        axes[i].set_yticks(range(len(sorted_idx)))
        axes[i].set_yticklabels(np.array(columns)[sorted_idx])
        axes[i].set_title(f'Fold {i+1}')
        axes[i].set_xticks([])
        i += 1

        # Hide the last axis
        axes[-1].set_visible(False)


def plot_results_folds(
        y_test_folds,
        y_preds_folds,
        y_preds_proba_folds,
        model_folds,
        columns_folds):
    
    plot_confusion_matrix_folds(
        y_test_folds=y_test_folds,
        y_preds_folds=y_preds_folds
    )
    plot_catboost_feature_importance_folds(
        model_folds=model_folds,
        columns_folds=columns_folds
    )