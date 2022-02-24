import numpy as np
import pandas as pd

import amex_cred_card_utils as autil
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import  RandomForestClassifier

# load the train data set and test data set
df_train = pd.read_csv(r'D:\github\DataScience\hacker-earth\amex_credit_card\train.csv')
df_test = pd.read_csv(r'D:\github\DataScience\hacker-earth\amex_credit_card\test.csv')

# drop columns  which are not fed in model
df_train.drop(['name'], axis=1, inplace=True)
df_test.drop(['name'], axis=1, inplace=True)
df_train.drop(['credit_limit'], axis=1, inplace=True)
df_test.drop(['credit_limit'], axis=1, inplace=True)

## Missing values handling

df_train = autil.fill_na_owns_car(df_train)
df_train = autil.fill_na_no_of_children(df_train)
df_train = autil.fill_na_no_of_days_employed(df_train)
df_train = autil.fill_na_total_family_members(df_train)
df_train = autil.fill_na_migrant_worker(df_train)
df_train = autil.fill_na_yearly_debt_payments(df_train)

df_test = autil.fill_na_owns_car(df_test)
df_test = autil.fill_na_no_of_children(df_test)
df_test = autil.fill_na_no_of_days_employed(df_test)
df_test = autil.fill_na_total_family_members(df_test)
df_test = autil.fill_na_migrant_worker(df_test)
df_test = autil.fill_na_yearly_debt_payments(df_test)

##predata processing - age comvert to range i.e. categorical variables
df_train = autil.age_to_range(df_train)
df_test = autil.age_to_range(df_test)
## we can now drop age as continuous variable
df_train.drop(['age'], axis=1, inplace=True)
df_test.drop(['age'], axis=1, inplace=True)
# convert age  to categorical variable encoder
df_train = autil.lb_encode_age_group(df_train)
df_test = autil.lb_encode_age_group(df_test)

# dummy candidate gender ,owns_car ,owns_house
# columns = ['gender', 'owns_car', 'owns_house']
df_train = pd.get_dummies(data=df_train, columns=['gender'])
df_train = pd.get_dummies(data=df_train, columns=['owns_car'])
df_train = pd.get_dummies(data=df_train, columns=['owns_house'])

df_test = pd.get_dummies(data=df_test, columns=['gender'])
df_test = pd.get_dummies(data=df_test, columns=['owns_car'])
df_test = pd.get_dummies(data=df_test, columns=['owns_house'])

## standardize net_yearly_income
df_train = autil.sd_net_yearly_income(df_train)
df_test = autil.sd_net_yearly_income(df_test)

# standardize no_of_days_employed
df_train = autil.sd_no_of_days_employed(df_train)
df_test = autil.sd_no_of_days_employed(df_test)

# standardize occupation_type -  label-encoder
df_train = autil.lb_encode_occupation_type(df_train)
df_test = autil.lb_encode_occupation_type(df_test)

# standardize total_family_members - no futher processing required apart from missing data handling
# migrant_worker -  no futher processing required apart from missing data handling
# standardize yearly_debt_payments
df_train = autil.sd_yearly_debt_payments(df_train)
df_test = autil.sd_yearly_debt_payments(df_test)

# credit_limit_used(%)  - rename the column for better readability
df_train.rename(columns={'credit_limit_used(%)': 'credit_limit_used'}, inplace=True)
df_test.rename(columns={'credit_limit_used(%)': 'credit_limit_used'}, inplace=True)

# standardize credit_score
df_train = autil.sd_credit_score(df_train)
df_test = autil.sd_credit_score(df_test)

# model - XGB Classifier
# df_train.columns

train_x = df_train.loc[:, ~df_train.columns.isin(['credit_card_default', 'customer_id'])]
# train_x = df_train.loc[:, df_train.columns != 'customer_id']
train_y = df_train['credit_card_default']
# xgb model
# model = XGBClassifier(n_estimators=100, max_depth=9, seed=2017)
# model = LogisticRegression()
train_x = train_x.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=2017)
model = RandomForestClassifier()
model.fit(train_x, train_y)
y_pred = model.predict(train_x)
cm = confusion_matrix(train_y, y_pred)
print(cm)

# generate class probabilities
# Notice that 2 elements will be returned in probs array,
# 1st element is probability for negative class,
# 2nd element gives probability for positive class
probs = model.predict_proba(train_x)
y_pred_prob = probs[:, 1]

# generate evaluation metrics
print("Accuracy:", metrics.accuracy_score(train_y, y_pred))
print("AUC :", metrics.roc_auc_score(train_y, y_pred_prob))

# extract false positive, true positive rate
fpr, tpr, thresholds = metrics.roc_curve(train_y, y_pred_prob)
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

##prediction on test data
# df_test.columns

test_x = df_test.loc[:, ~df_test.columns.isin(['customer_id'])]
test_x = test_x.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
pred_test_y = model.predict(test_x)
df_test['credit_card_default'] = pred_test_y

header = ["customer_id", "credit_card_default"]
df_test.to_csv(r"D:\github\DataScience\hacker-earth\amex_credit_card\random_forest.csv", columns=header, index=False)
