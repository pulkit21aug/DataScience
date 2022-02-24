import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import  metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#load the train data set and test data set
df_train = pd.read_csv(r'E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\train.csv')
df_test = pd.read_csv(r'E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\test.csv')

#split dataset into train-test

x =  df_train.loc[:, ~df_train.columns.isin(['Severity', 'Accident_ID','Violations','Cabin_Temperature','Accident_Type_Code','Max_Elevation','Adverse_Weather_Metric'])]
y =  df_train.loc[:, df_train.columns == 'Severity']
df_test_x =  df_test.loc[:, ~df_test.columns.isin(['Accident_ID','Violations','Cabin_Temperature','Accident_Type_Code','Max_Elevation','Adverse_Weather_Metric'])]

x_train = x
y_train = y

#transform Safety_Score
mms = MinMaxScaler()
#mms = StandardScaler()

#transform Safety_Score
x_train['Safety_Score'] = mms.fit_transform(x_train[['Safety_Score']])
df_test_x['Safety_Score'] = mms.transform(df_test_x[['Safety_Score']])

#transform Days_Since_Inspection
x_train['Days_Since_Inspection'] = mms.fit_transform(x_train[['Days_Since_Inspection']])
df_test_x['Days_Since_Inspection'] = mms.transform(df_test_x[['Days_Since_Inspection']])

#transform Total_Safety_Complaints
x_train['Total_Safety_Complaints'] = mms.fit_transform(x_train[['Total_Safety_Complaints']])
df_test_x['Total_Safety_Complaints'] = mms.transform(df_test_x[['Total_Safety_Complaints']])

#transform Control_Metric
x_train['Control_Metric'] = mms.fit_transform(x_train[['Control_Metric']])
df_test_x['Control_Metric'] = mms.transform(df_test_x[['Control_Metric']])

#transform Cabin_Temperature
# x_train['Cabin_Temperature'] = mms.fit_transform(x_train[['Cabin_Temperature']])
# df_test_x['Cabin_Temperature'] = mms.transform(df_test_x[['Cabin_Temperature']])

#transform Max_Elevation
# x_train['Max_Elevation'] = mms.fit_transform(x_train[['Max_Elevation']])
# df_test_x['Max_Elevation'] = mms.transform(df_test_x[['Max_Elevation']])

#transform Accident_Type_Code
# x_train['Accident_Type_Code'] = x_train['Accident_Type_Code'].astype('category')
# accident_type_code_dummies = pd.get_dummies(x_train.Accident_Type_Code,prefix="accident_type_code")
# x_train = pd.concat([x_train, accident_type_code_dummies], axis=1)
# x_train.drop(['Accident_Type_Code'],axis=1,inplace=True)
#
# df_test_x['Accident_Type_Code'] = df_test_x['Accident_Type_Code'].astype('category')
# act_dummies = pd.get_dummies(df_test_x.Accident_Type_Code,prefix="accident_type_code")
# df_test_x = pd.concat([df_test_x, act_dummies], axis=1)
# df_test_x.drop(['Accident_Type_Code'],axis=1,inplace=True)

#Take violations as dummies
# x_train['Violations'] = x_train['Violations'].astype('category')
# Violations_dummies = pd.get_dummies(x_train.Violations,prefix="Violations")
# x_train = pd.concat([x_train, Violations_dummies], axis=1)
# x_train.drop(['Violations'],axis=1,inplace=True)
#
# df_test_x['Violations'] = df_test_x['Violations'].astype('category')
# Violations_dummies_test = pd.get_dummies(df_test_x.Violations,prefix="Violations")
# df_test_x = pd.concat([df_test_x, Violations_dummies_test], axis=1)
# df_test_x.drop(['Violations'],axis=1,inplace=True)


#random forest
rfc_model =  RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 42,max_depth=None)
rfc_model.fit(x_train, y_train)
y_pred_rfc = rfc_model.predict(x_train)
print(metrics.confusion_matrix(y_train, y_pred_rfc))
print(metrics.classification_report(y_train, y_pred_rfc))

################################## xgBoost model  ########################################
xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
#xgb_model = xgb.XGBClassifier(objective="multi:softmax", random_state=42,num_class=4)
xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_train)
print(metrics.confusion_matrix(y_train, y_pred))
print(metrics.classification_report(y_train, y_pred))


#SVM
from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(x_train, y_train)
y_pred_svm = rfc_model.predict(x_train)
print(metrics.confusion_matrix(y_train, y_pred_svm))
print(metrics.classification_report(y_train, y_pred_svm))

################### Prediction for test data final dataset ###################################

y_pred_results = rfc_model.predict(df_test_x)
df_test['Severity'] = y_pred_results
header = ["Accident_ID", "Severity"]
df_test.to_csv(r"E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\results-random.csv", columns = header,index=False)
#RF is best

y_pred_results = svm_model.predict(df_test_x)
df_test['Severity'] = y_pred_results
header = ["Accident_ID", "Severity"]
df_test.to_csv(r"E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\results-svm.csv", columns = header,index=False)
#SVM is worst

y_pred_results_xgb = xgb_model.predict(df_test_x)
df_test['Severity'] = y_pred_results_xgb
header = ["Accident_ID", "Severity"]
df_test.to_csv(r"E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\results-xgb.csv", columns = header,index=False)
#xgb model less than RF but far better than SVM