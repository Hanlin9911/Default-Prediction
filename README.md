# Customer Default Prediction
This project aims to analyze and predict custoemr default in presumably UK. The dataset contains various customer information and loan information of 32586 individuals. 
Check [Loan Default Prediction](https://www.kaggle.com/datasets/prakashraushan/loan-dataset) for more detailed information. 

This dataset contains 12 columns, including `Current_loan_status` as the target column. The remaining 11 columns can be divided into 2 categories:
1. Customer Label - `customer_id`, `customer_age`, `customer_income`, `home_ownership`, `employment_duration`, `historical_default`, `cred_hist_length`
2. Loan Label - `loan_intent`, `loan_grade`, `loan_int_rate`, `term_years`

## Data Validation and Data Cleaning
`customer_age` was cross-validated with `cred_hist_length` and `employment_duration`. 
Instances with `customer_age` greater than 80 or smaller than 18 were dropped as well as the instances with the start-working-age smaller than 13 since these entries were considered as invalid.

Missing values in `historical_default` was replaced by "N" but later it was found that this is not the best approach in this case. Due to the time restrain, other approaches were not explored.   
Missing valuse in `loan_int_rate` was imputed using the sample mean. 
Missing values in `employment_duration` was dropped or encoded by weight of evidence encoder.
**See more on [Weight of Evidence Encoding](https://ishanjainoffical.medium.com/understanding-weight-of-evidence-woe-with-python-code-cd0df0e4001e) and [Optimal Binning](http://gnpalencia.org/optbinning/tutorials/tutorial_binary.html).**

## Exploratory Data Analysis (EDA)
For numeric features, the **Gini coefficient** was used to compare the correlation between the features and the target. It was shown in the Disucssion section that the correlation discovered here was not linear. 

## Baseline Model
Based on the established gini coefficient table, `customer_income`, `loan_int_rate` and `loan_grade` were chosen to establish the baseline model using KNearestNeigbor and RandomForestClassifier. 
The WOE encoder was applied to compare wheather the choice of different encoding methods/different missing value handling methods would have an effect on model performance.  
However, the result was not as satisfying. All models had an accuracy ratio around 0.8 with a relatively low recall rate, especially when the dataset was encoded by WOE (including missing values in `employment_duration`).  
This suggested that incorporating the missing values in `employment_duration` may have increased the difficulty/noise in terms of target prediction. In addition, more features needs to be included in order to improve the model performance. 

## RMF with all features
With the conclusion above, a RMF model that took all features was established to test the potential maximum performance. Then, the feature importance was pulled out to faciliate feature selection in order to increase the model efficiency.  
By accident, in one test run, missing values in `historical_default` were not replaced by "N", and that model had the best performance with 0.97 accuracy ratio. This case was further discussed in the Discussion section. After replacing NaN in `historical_default` by "N", the accuracy dropped to 0.93 with a relatively low recall rate.  
The features with the most predictive power based on feature importance was identified - `customer_income`, `loan_int_rate`, `loan_amnt` and `employment_duration`. At the same time, the gini coefficient for these four features are high. Surprisingly, `loan_grade` had less predictive power than expected. `home_ownership` and `loan_intent` would be included in the future models. 

## Model 2 - with 6 features
The RandomForestClassifier had a better performance compared with KNN. However, both models had relatively low recall rate. Therefore, the RandomForestClassifier with  `customer_income`, `loan_int_rate`, `loan_amnt` and `employment_duration`, `loan_intent`, and `home_ownership` as features was chosen as the final model. 
