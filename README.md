# Discovering Predictive Features for Clinically Definite Multiple Sclerosis from Clinical Isolated Syndromes with Machine Learning

This repository houses the code and data for the study titled: **Discovering Predictive Features for Clinically Definite Multiple Sclerosis from Clinical Isolated Syndromes with Machine Learning**.

- [Paper (TBA)]()

## Overview

This study leverages advanced machine learning models to predict the progression from clinically isolated syndromes (CIS) to clinically definite multiple sclerosis (CDMS). We explored several models, including CatBoost, XGBoost, LightGBM, Random Forest, Support Vector Machine, and Logistic Regression, to identify key features that signal a higher risk of developing CDMS.

### Key Highlights

- **Model Performance:** 
  - CatBoost achieved an AUC of 0.93, demonstrating high predictive accuracy.
  - XGBoost closely followed with an AUC of 0.9202.
  
- **Significant Predictors:** 
  - Consistent key predictors identified across all models include:
    - Periventricular_MRI
    - Infratentorial_MRI
    - Oligoclonal_Bands
    - Schooling
    - Symptom_Motor
    
- **Clinical Implications:** 
  - The findings underscore the potential of machine learning as a tool for early identification of high-risk CIS patients, potentially guiding early intervention strategies to improve patient outcomes.

## Repository Structure

- `data/`: Contains raw and split data files.
- `notebooks/`: Jupyter Notebooks for data exploration, model training, and analysis.
- `src/`: Python scripts for preprocessing, feature engineering, and model evaluation.
- `results/`: Contains output from model evaluations, including metrics and visualizations.
- `Pipfile`: Dependency management file for setting up the project environment.
- `Pipfile.lock`: Lock file to ensure a consistent environment across different setups.

## Setup Instructions

Follow these steps to set up the project environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/luumsk/CIStoCDMS.git
   ```

2. **Navigate to the repository directory:**
   ```bash
   cd CIStoCDMS
   ```

3. **Create a virtual environment using Pipenv:**
   ```bash
   pipenv install
   ```

4. **Activate the virtual environment:**
   ```bash
   pipenv shell
   ```

5. **Run the Jupyter Notebooks:**
   ```bash
   jupyter notebook <notebook_filename>.ipynb
   ```


We welcome contributions! Please feel free to open issues or submit pull requests. For any questions or support, contact us at `khue.luu@g.nsu.ru`.

