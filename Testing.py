from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation imports
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
    accuracy_score,
    make_scorer
)


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt



def cat_visual(df, col, ax):
    '''
    This function aims to visualize the proportion of Default vs No Default in different categories of a chosen column

    :param [df]: dataframe of interest
    :type [df]: pd.DataFrame

    :param [col]: columnn name of interest
    :type [col]: str

    :param [ax]: the axis to be plotted on
    :type [ax]: matplotlib.axe.Axes
    ...
    :return: the summary of DEFAULT proportion grouped by selected column
    :rtype: pd.DataFrame
    
    '''
    summary = (df.groupby(col,as_index=False)['Current_loan_status'].value_counts(normalize=True))

    sns.barplot(data=summary, x=col, y='proportion', hue='Current_loan_status',ax=ax)
    ax.set_title(f'Current Loan Status Percentage by {col}')
    ax.set_ylabel('Proportion')

    return summary


# GridSearch/Hyperparameter Tuning
def weighted_recall_precision(y_true, y_pred): # From ChatGPT, but parameter is chosen after testing.
    recall = recall_score(y_true, y_pred, pos_label=1)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy


# According to a search
# AR > 0.85, and precision and recall both > 0.7 is considered as a good model. 
# However... few model tested in this project met the 0.8 cut-line. 

def param_tuning(X_tr, y_tr, model, param, cat_col=[], num_col=[]):
    # Doc string is writtne by ChatGPT... Thanks to ChatGPT
    # But I wrote all the functions! except for the weighted_recall_precision....
    """
    Perform hyperparameter tuning for a given model using GridSearchCV with custom scoring.

    Parameters:
    ----------
    X_tr : pandas.DataFrame
        Training feature data.
    y_tr : pandas.Series or numpy.ndarray
        Training target data.
    model : sklearn.base.BaseEstimator
        The model class (e.g., LogisticRegression, RandomForestClassifier) to be tuned.
    param : dict
        Dictionary with parameters names (str) as keys and lists of parameter settings to try as values, 
        defining the grid of hyperparameters to search.
    cat_col : list of str, optional
        List of column names corresponding to categorical features. Default is an empty list.
    num_col : list of str, optional
        List of column names corresponding to numerical features. Default is an empty list.

    Returns:
    -------
    preprocessor_t : sklearn.compose.ColumnTransformer
        The preprocessor transformer used for the best model found during GridSearchCV.
    
    Notes:
    -----
    - The function sets up a pipeline for preprocessing the data, which includes one-hot encoding for 
      categorical features and scaling for numerical features.
    - A custom scoring function (`weighted_recall_precision`) is used that prioritizes recall and accuracy 
      with specific conditions.
    - The function prints the best parameters found during GridSearchCV.
    
    Example:
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> param = {'classifier__C': [0.1, 1, 10], 'classifier__penalty': ['l2']}
    >>> best_preprocessor = param_tuning(X_train, y_train, LogisticRegression(), param, cat_col=['cat_feature'], num_col=['num_feature'])
    >>> print(best_preprocessor)
    """


    # Define steps
    
    # Encoding
    categorical_transformer = Pipeline(
        steps=[
            ('encoder', OneHotEncoder(drop='first',sparse_output=False)),
               ("scaler", StandardScaler())
               ]
    )
    # Scaling
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])


    # no categorical feature
    if len(cat_col) == 0:
        # Scaling only
        preprocessor_t = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_col)
            ]
        )





    # no numeric feature
    elif len(num_col) == 0:
        # encoding only
        preprocessor_t = ColumnTransformer(
                            transformers=[
                                ('cat', categorical_transformer, cat_col)
                            ]
        )





    # categorical feature and numeric feature both present
    else:

         # Encoding and scaling
         preprocessor_t = ColumnTransformer(
                            transformers=[
                                ('cat', categorical_transformer, cat_col),
                                ('num', numeric_transformer, num_col)
            ]
        )





    # Instance of model
    model = Pipeline(
        steps=[("preprocessor_t", preprocessor_t), ('classifier', model)]
                    )
    
    # Define the custom scorer
    custom_scorer = make_scorer(weighted_recall_precision)
    grid_search = GridSearchCV(estimator=model, param_grid=param, cv=5, scoring=custom_scorer)

    # Gridsearch
    grid_search.fit(X_tr, y_tr)

    print("Best parameters found: ", grid_search.best_params_)

    return preprocessor_t



# Testing
def testing(X_tr, y_tr, X_te, y_te, model, preprocessor):
  '''
  Perform training and testing for a given model with optimized hyperparamters. 

  Parameters:
  ----------
  X_tr : pandas.DataFrame
      Training feature data.
  y_tr : pandas.Series or numpy.ndarray
      Training target data.
  X_te : pandas.DataFrame
      Testing feature data.
  y_te : pandas.Series or numpy.ndarray
      Testing target data.
  model : sklearn.base.BaseEstimator
      The model class with tuned hyperparameter(e.g., LogisticRegression, RandomForestClassifier) to be trained and tested.
  preprocessor_t : sklearn.compose.ColumnTransformer
    The preprocessor transformer used for the best model found during GridSearchCV.
  
  Returns:
  -------
  y_pr : pandas.DataFrame
      Predicted target values.

  '''
  model_f = Pipeline(
  steps=[("preprocessor_t", preprocessor), ('classifier', model)])

  model_f.fit(X_tr, y_tr)
  y_pr = model_f.predict(X_te)

  return y_pr


# Evaluation
def evaluating(y_pr,y_te):
    '''
    The function displays a confusion matrix and prints a classification report with given data.

    Parameters:
    ----------
    y_pr : pandas.Series or numpy.ndarray
        Predicted target data.
    y_te : pandas.Series or numpy.ndarray
        Testing target data.
    '''

    # Label names for confusion matrix
    labels = np.array([0, 1], dtype=np.int64)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_te, y_pr), display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    print(classification_report(y_te, y_pr))




