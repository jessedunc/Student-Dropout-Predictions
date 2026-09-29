# Student Dropout Predictions

## Project Overview

The goal of this project was to compare three machine learning algorithms, KNN, logistic regression, and random forest, performance at predicting whether a university student will dropout,
graduate, or remain enrolled before and after hyperparameter tuning and to use learning curves to examine how each algorithms perform based on the size of the data. 

The data used for this project contained 36 academic, demagrophic, and economic features for 4,424 university students enrolled between 2008/2009 and 2018/2019. The target attribute contained one 
of three values, Dropout, Graduate, and Enrolled, to indicate the students outcome. The dataset was originally  created for the study "Predicting Student Dropout and Academic Success".
The data set can be downloaded at https://www.kaggle.com/datasets/adilshamim8/predict-students-dropout-and-academic-success

As part of the preprocessing numeric attributes were scaled using`StandardScaler()`  while categorical attributes were OneHotEncoded. Each model included 8 features selected based on mutual
information. Hyperparameter tuning was done using `GridSearchCV()` optimized for F1 Macro due to the imbalance of the target class.


## Project Results

A table indicating the results for each algorithm before and after hyperparameter tuning can be seen below

Model Comparison
| Algorithm           | Baseline Accuracy | Tuned Accuracy | Baseline F1 Macro | Tuned F1 Macro |
| ------------------- | ----------------- | -------------- | ----------------- | -------------- |
| KNN                 | 0.69              | 0.71           | 0.63              | 0.63           |
| Logistic Regression | 0.75              | 0.75           | 0.66              | 0.66           |
| Random Forest       | 0.73              | 0.75           | 0.64              | 0.66           |


The results show us the KNN model performed the worst of the three while the logistic regression and random forest models performed similarly. Hyperparameter tuning had the greatest effect on 
the random forest model with an increase of about 0.02 for both accuracy and F1 Macro, a slight effect on the KNN model, and a negligible effect on the logistic regression model. The random 
forest model was ultimetly recommended because of all the models it achieved the highest recall for the Dropout class. Which means the model had the lowest false-negatives, which in this case
is more important than minimizing false positives because finding all students who are at risk of dropping out is more important than ensuring all positive predictions of dropouts are accurate. 

## Key Findings 

All three models performed very poorly at predicting the Enrolled class with an average f1-score of 0.36 for the tuned class. This is likely due to the fact that the Enrolled class is a timing snapshot
rather than a completed trajectory. Upon further investigation of the missclassifications of the baseline logistic regression model, it was found that Enrolled students misclassified as Dropout students 
had lower 1st and 2nd semester grades and lower 1st and 2nd semester approved credits, features shown to have the highest mutual information with the target class, than Enrolled students misclassified
as Graduate students. This indicates that the model was misclassifying Enrolled students based on their final trajectory.  

## How to Run

1. Clone this repository:
```bash
   git clone https://github.com/jessedunc/Student-Dropout-Predictions
   cd Student-Dropout-Predictions
```

2. Install the required packages:
```bash
   pip install -r requirements.txt
```

3. Open and run the notebook:
```bash
   jupyter notebook student_predictions_notebook.ipynb
```
   Run all cells in order from top to bottom.