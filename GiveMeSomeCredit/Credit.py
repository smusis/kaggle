import pandas as pd
import numpy as np
from sklearn import (metrics, cross_validation, linear_model, preprocessing, ensemble)

SEED = 42 

#Read training data
dftrain = pd.read_csv('cs-training.csv')

#Remove the row with age 0
dftrain = dftrain[(dftrain['age'] != 0)]

#Remove values 96 & 98 from "NumberOfTime30-59DaysPastDueNotWorse"
dftrain = dftrain[(dftrain['NumberOfTime30-59DaysPastDueNotWorse'] != 96) & (dftrain['NumberOfTime30-59DaysPastDueNotWorse'] != 98)]

#Remove values 96 & 98 from "NumberOfTime60-89DaysPastDueNotWorse"
dftrain = dftrain[(dftrain['NumberOfTime60-89DaysPastDueNotWorse'] != 96) & (dftrain['NumberOfTime60-89DaysPastDueNotWorse'] != 98)]

#Remove values 96 & 98 from "NumberOfTimes90DaysLate"
dftrain = dftrain[(dftrain['NumberOfTimes90DaysLate'] != 96) & (dftrain['NumberOfTimes90DaysLate'] != 98)]

#Impute 'NumberOfDependents' with 0
dftrain['NumberOfDependents'].fillna(0, inplace=True)

#MonthlyIncome imputation with 0
dftrain['MonthlyIncome'].fillna(0, inplace=True)

#Filter DebtRatio
dftrain = dftrain[(dftrain['DebtRatio'] < 15000)]


cols_to_keep = ['RevolvingUtilizationOfUnsecuredLines' , 'age' , 'NumberOfTime30-59DaysPastDueNotWorse' , 'DebtRatio' , 'MonthlyIncome' , 'NumberOfOpenCreditLinesAndLoans' , 'NumberOfTimes90DaysLate' , 'NumberRealEstateLoansOrLines' , 'NumberOfTime60-89DaysPastDueNotWorse' , 'NumberOfDependents']

X_train1 = dftrain[cols_to_keep]
y_train1 = dftrain['SeriousDlqin2yrs']


#Random Forest Classifier
clf = ensemble.RandomForestClassifier(n_estimators=1000,min_samples_split=500)


#Read Test Data
dftest = pd.read_csv('cs-test.csv',usecols=cols_to_keep)

#Impute 'NumberOfDependents' with 0
dftest['NumberOfDependents'].fillna(0, inplace=True)

#MonthlyIncome imputation with 0
dftest['MonthlyIncome'].fillna(0, inplace=True)



p_test = dftest[cols_to_keep]


mean_auc = 0.0
n = 4  # repeat the CV procedure 4 times to get more precise results
for i in range(n):
    # for each iteration, select 20% of the data for cross-validation
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
        X_train1, y_train1, test_size=.20, random_state=i*SEED)

    # train model and make predictions
    clf.fit(X_train, y_train) 
    preds = clf.predict_proba(X_cv)[:, 1]

    # compute AUC metric
    fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
    roc_auc = metrics.auc(fpr, tpr)
    mean_auc += roc_auc

#Mean AUC for all the folds
print("Mean AUC: %f" % (mean_auc/n))

#Save results as CSV
def save_csv(predictions, filename):
    """Save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("Id,Probability\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))
			

#Train the model			
clf.fit(X_train1, y_train1)

#Prediction on test data
probs = clf.predict_proba(p_test)[:, 1]

save_csv(probs, "OutputCredit.csv")