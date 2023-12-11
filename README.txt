Capstone Project: VEHICLE LOAN DEFAULT PREDICTION 

This project is to predict vehicle loan default. The dataset was taken from [Kaggle](https://www.kaggle.com/datasets/avikpaul4u/vehicle-loan-default-prediction?select=train.csv/).

The dataset is imbalanced and was trained on 4 models, namely Logistic Regression, Decision Tree, Random Forest and XGBoost. The model outputs indicate XGBoost as the optimal model due to its highest F1 score of 0.4 on the test set. The two most highly predictive features are state id and credit score, which is consistent with EDA findings. It is recommended to request 84002 customers whom the model predicts to have defaulted but actually not to mortgage their security against the loan as they are very likely to default in the future.
