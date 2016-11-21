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

#RevolvingUtilizationOfUnsecuredLines remove more than 3
dftrain = dftrain[(dftrain['RevolvingUtilizationOfUnsecuredLines'] < 4)]

#Filter DebtRatio
dftrain = dftrain[(dftrain['DebtRatio'] < 15000)]

#Adding new features
#Add NumberOfTime30-59DaysPastDueNotWorse'] and 'NumberOfTime60-89DaysPastDueNotWorse'
dftrain['NumberOfTime30-89DaysPastDueNotWorse'] = dftrain['NumberOfTime30-59DaysPastDueNotWorse'] + dftrain['NumberOfTime60-89DaysPastDueNotWorse']

#Add NumberOfTimePastDue adding all three past dues
dftrain['NumberOfTimePastDue'] = dftrain['NumberOfTime30-59DaysPastDueNotWorse'] + dftrain['NumberOfTime60-89DaysPastDueNotWorse'] + dftrain['NumberOfTimes90DaysLate']

#Never Due
dftrain['NeverDue'] = ((dftrain['NumberOfTime30-59DaysPastDueNotWorse'] < 1).astype(int) * (dftrain['NumberOfTime60-89DaysPastDueNotWorse'] < 1).astype(int) * (dftrain['NumberOfTimes90DaysLate'] < 1).astype(int))

#Does the person has multiple real estate loans
dftrain['HasRealestateLoan'] = (dftrain['NumberRealEstateLoansOrLines'] > 0).astype(int)
dftrain['MultipleLoans'] = (dftrain['NumberRealEstateLoansOrLines'] > 1).astype(int)

#HasDependents & LogDependents
dftrain['HasDependents'] = (dftrain['NumberOfDependents'] > 0).astype(int)
dftrain['LogDependents'] = np.log(dftrain['NumberOfDependents'])
dftrain['LogDependents'] = dftrain['LogDependents'].replace([np.inf, -np.inf], 0)
dftrain['LogDependents'].fillna(0, inplace=True)

#Log Income
dftrain['LogMonthlyIncome'] = np.log(dftrain['MonthlyIncome'])
dftrain['LogMonthlyIncome'] = dftrain['LogMonthlyIncome'].replace([np.inf, -np.inf], 0)
dftrain['LogMonthlyIncome'].fillna(0, inplace=True)

#IncomePerPerson
dftrain['IncomePerPerson'] = (dftrain['MonthlyIncome'] / dftrain['NumberOfDependents'])
dftrain['IncomePerPerson'] = dftrain['IncomePerPerson'].replace([np.inf, -np.inf], 0)
dftrain['IncomePerPerson'].fillna(0, inplace=True)

#LogMonthlyIncomePerPerson
dftrain['LogMonthlyIncomePerPerson'] = (dftrain['LogMonthlyIncome'] - np.log(dftrain['NumberOfDependents']))
dftrain['LogMonthlyIncomePerPerson'] = dftrain['LogMonthlyIncomePerPerson'].replace([np.inf, -np.inf], 0)
dftrain['LogMonthlyIncomePerPerson'].fillna(0, inplace=True)

#LogDebtRatio
dftrain['LogDebtRatio'] = np.log(dftrain['DebtRatio'])
dftrain['LogDebtRatio'] = dftrain['LogDebtRatio'].replace([np.inf, -np.inf], 0)
dftrain['LogDebtRatio'].fillna(0, inplace=True)



cols_to_keep = ['RevolvingUtilizationOfUnsecuredLines' , 'age' , 'NumberOfTime30-59DaysPastDueNotWorse' , 'DebtRatio' , 'MonthlyIncome' , 'NumberOfOpenCreditLinesAndLoans' , 'NumberOfTimes90DaysLate' , 'NumberRealEstateLoansOrLines' , 'NumberOfTime60-89DaysPastDueNotWorse' , 'NumberOfDependents','NumberOfTime30-89DaysPastDueNotWorse','NumberOfTimePastDue','NeverDue','HasRealestateLoan','MultipleLoans','HasDependents','LogDependents'
,'LogMonthlyIncome','IncomePerPerson','LogMonthlyIncomePerPerson','LogDebtRatio']

X_train1 = dftrain[cols_to_keep]
y_train1 = dftrain['SeriousDlqin2yrs']


#Random Forest Classifier
clf = ensemble.RandomForestClassifier(n_estimators=1000,min_samples_split=1000)

#Resd these columns from test data
cols_to_read = ['RevolvingUtilizationOfUnsecuredLines' , 'age' , 'NumberOfTime30-59DaysPastDueNotWorse' , 'DebtRatio' , 'MonthlyIncome' , 'NumberOfOpenCreditLinesAndLoans' , 'NumberOfTimes90DaysLate' , 'NumberRealEstateLoansOrLines' , 'NumberOfTime60-89DaysPastDueNotWorse' , 'NumberOfDependents']


#Read Test Data
dftest = pd.read_csv('cs-test.csv',usecols=cols_to_read)

#Impute 'NumberOfDependents' with 0
dftest['NumberOfDependents'].fillna(0, inplace=True)

#MonthlyIncome imputation with 0
dftest['MonthlyIncome'].fillna(0, inplace=True)


#Add NumberOfTime30-59DaysPastDueNotWorse'] and 'NumberOfTime60-89DaysPastDueNotWorse'
dftest['NumberOfTime30-89DaysPastDueNotWorse'] = dftest['NumberOfTime30-59DaysPastDueNotWorse'] + dftest['NumberOfTime60-89DaysPastDueNotWorse']

#Add NumberOfTimePastDue adding all three past dues
dftest['NumberOfTimePastDue'] = dftest['NumberOfTime30-59DaysPastDueNotWorse'] + dftest['NumberOfTime60-89DaysPastDueNotWorse'] + dftest['NumberOfTimes90DaysLate']

#Never Due
dftest['NeverDue'] = ((dftest['NumberOfTime30-59DaysPastDueNotWorse'] < 1).astype(int) * (dftest['NumberOfTime60-89DaysPastDueNotWorse'] < 1).astype(int) * (dftest['NumberOfTimes90DaysLate'] < 1).astype(int))

#Does the person has multiple real estate loans
dftest['HasRealestateLoan'] = (dftest['NumberRealEstateLoansOrLines'] > 0).astype(int)
dftest['MultipleLoans'] = (dftest['NumberRealEstateLoansOrLines'] > 1).astype(int)

#HasDependents & LogDependents
dftest['HasDependents'] = (dftest['NumberOfDependents'] > 0).astype(int)
dftest['LogDependents'] = np.log(dftest['NumberOfDependents'])
dftest['LogDependents'] = dftest['LogDependents'].replace([np.inf, -np.inf], 0)
dftest['LogDependents'].fillna(0, inplace=True)

#Log Income
dftest['LogMonthlyIncome'] = np.log(dftest['MonthlyIncome'])
dftest['LogMonthlyIncome'] = dftest['LogMonthlyIncome'].replace([np.inf, -np.inf], 0)
dftest['LogMonthlyIncome'].fillna(0, inplace=True)

#IncomePerPerson
dftest['IncomePerPerson'] = (dftest['MonthlyIncome'] / dftest['NumberOfDependents'])
dftest['IncomePerPerson'] = dftest['IncomePerPerson'].replace([np.inf, -np.inf], 0)
dftest['IncomePerPerson'].fillna(0, inplace=True)

#LogMonthlyIncomePerPerson
dftest['LogMonthlyIncomePerPerson'] = (dftest['LogMonthlyIncome'] - np.log(dftest['NumberOfDependents']))
dftest['LogMonthlyIncomePerPerson'] = dftest['LogMonthlyIncomePerPerson'].replace([np.inf, -np.inf], 0)
dftest['LogMonthlyIncomePerPerson'].fillna(0, inplace=True)

#LogDebtRatio
dftest['LogDebtRatio'] = np.log(dftest['DebtRatio'])
dftest['LogDebtRatio'] = dftest['LogDebtRatio'].replace([np.inf, -np.inf], 0)
dftest['LogDebtRatio'].fillna(0, inplace=True)


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
