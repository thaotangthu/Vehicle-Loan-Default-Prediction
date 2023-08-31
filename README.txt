Capstone Project: VEHICLE LOAN DEFAULT PREDICTION 

This project is to predict vehicle loan default. The dataset was originally taken from [Kaggle](https://www.kaggle.com/datasets/avikpaul4u/vehicle-loan-default-prediction?select=train.csv)

The dataset is pretty imbalanced and was trained on 4 models, namely Logistic Regression, Decision Tree, Random Forest and XGBoost. The model outputs indicate XGBoost at threshold **0.51** as the optimal model due to its highest F1 score of 0.4 on test set. Also, two most important features that are highly predictive are state id and credit score, which matches with EDA findings.

There are 84002 customers whom the model predicts that they should have defaulted but actually not. This implies that they are very likely to default in the future. Therefore, it is recommended to ask them to mortgage some sort of security against their loan as soon as possible.
