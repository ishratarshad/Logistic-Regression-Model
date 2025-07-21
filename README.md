# Logistic Regression Model – Airbnb Superhost Classification

This project uses logistic regression to predict whether an Airbnb host is a superhost using preprocessed listing data. The process includes model training, evaluation, feature selection, and persistence.

## Dataset
- `data_LR/airbnbData_train.csv`: Preprocessed dataset with one-hot encoding, scaling, and imputed values.

## Techniques Used:
- Logistic Regression (scikit-learn)
- GridSearchCV for hyperparameter tuning
- Confusion matrix, ROC curve, AUC, and precision-recall curve evaluation
- Feature selection using SelectKBest
- Model persistence using `pickle`

## Model Goals
- Classify hosts as superhosts (`True` or `False`)
- Evaluate default vs tuned logistic regression models
- Analyze feature importance and compare AUC scores

## Files used:
- `ModelSelectionForLogisticRegression.ipynb`: The full notebook
- `model_best.pkl`: Pickled best model
- `data_LR/airbnbData_train.csv`: Cleaned Airbnb dataset

## Libraries
- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

> Developed as part of a machine learning lab exercise!
