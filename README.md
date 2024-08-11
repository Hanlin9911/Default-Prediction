# Customer Default Prediction
This project aims to analyze and predict custoemr default in presumably UK. The dataset contains various customer information and loan information of 32586 individuals. 
Check [Loan Default Prediction](https://www.kaggle.com/datasets/prakashraushan/loan-dataset) for more detailed information. 

This dataset contains 12 columns, including `Current_loan_status` as the target column. The remaining 11 columns can be divided into 2 categories:
1. Customer Label - `customer_id`, `customer_age`, `customer_income`, `home_ownership`, `employment_duration`, `historical_default`, `cred_hist_length`
2. Loan Label - `loan_intent`, `loan_grade`, `loan_int_rate`, `term_years`

## Data Validation and Data Cleaning
`customer_age` was cross-validated with `cred_hist_length` and `employment_duration`. 
Instances with `customer_age` greater than 80 or saller than 18 were dropped as well as the instances with the start-working-age smaller than 13 since these entries were considered as invalid.

Missing values in `historical_default` was replaced by "N" but later it was found that this is not the best approach in this case. Due to the time restrain, other approaches were not explored.   
Missing valuse in `loan_int_rate` was imputed using the sample mean. 
Missing values in `employment_duration` was dropped or encoded by weight of evidence encoder.
**See more on [Weight of Evidence Encoding](https://ishanjainoffical.medium.com/understanding-weight-of-evidence-woe-with-python-code-cd0df0e4001e) and [Optimal Binning](http://gnpalencia.org/optbinning/tutorials/tutorial_binary.html).**

## Exploratory Data Analysis (EDA)
For numeric features, the **Gini coefficient** was used to compare the correlation between the features and the target. It was shown in the Disucssion session that the correlation discovered here was not linear. 

## Baseline Model
Based on the established gini coefficient table, `customer_income`, `loan_int_rate` and `loan_grade` were chosen to establish the baseline model using KNearestNeigbor and RandomForestClassifier. 
