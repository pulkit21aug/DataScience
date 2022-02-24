import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#load the train data set and test data set
df_train = pd.read_csv(r'E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\train.csv')
df_test = pd.read_csv(r'E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\test.csv')

#split dataset into train-test

x =  df_train.loc[:, ~df_train.columns.isin(['Severity', 'Accident_ID'])]
y =  df_train.loc[:, df_train.columns == 'Severity']
df_test_x =  df_test.loc[:, ~df_test.columns.isin(['Accident_ID'])]

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=.3,random_state=42)


#transform Safety_Score
mms = MinMaxScaler()
#mms = StandardScaler()

#transform Safety_Score
x_train['Safety_Score'] = mms.fit_transform(x_train[['Safety_Score']])
x_val['Safety_Score'] = mms.transform(x_val[['Safety_Score']])
df_test_x['Safety_Score'] = mms.transform(df_test_x[['Safety_Score']])

#transform Days_Since_Inspection
x_train['Days_Since_Inspection'] = mms.fit_transform(x_train[['Days_Since_Inspection']])
x_val['Days_Since_Inspection'] = mms.transform(x_val[['Days_Since_Inspection']])
df_test_x['Days_Since_Inspection'] = mms.transform(df_test_x[['Days_Since_Inspection']])

#transform Total_Safety_Complaints
x_train['Total_Safety_Complaints'] = mms.fit_transform(x_train[['Total_Safety_Complaints']])
x_val['Total_Safety_Complaints'] = mms.transform(x_val[['Total_Safety_Complaints']])
df_test_x['Total_Safety_Complaints'] = mms.transform(df_test_x[['Total_Safety_Complaints']])

#transform Control_Metric
x_train['Control_Metric'] = mms.fit_transform(x_train[['Control_Metric']])
x_val['Control_Metric'] = mms.transform(x_val[['Control_Metric']])
df_test_x['Control_Metric'] = mms.transform(df_test_x[['Control_Metric']])

#transform Cabin_Temperature
x_train['Cabin_Temperature'] = mms.fit_transform(x_train[['Cabin_Temperature']])
x_val['Cabin_Temperature'] = mms.transform(x_val[['Cabin_Temperature']])
df_test_x['Cabin_Temperature'] = mms.transform(df_test_x[['Cabin_Temperature']])

#transform Max_Elevation
x_train['Max_Elevation'] = mms.fit_transform(x_train[['Max_Elevation']])
x_val['Max_Elevation'] = mms.transform(x_val[['Max_Elevation']])
df_test_x['Max_Elevation'] = mms.transform(df_test_x[['Max_Elevation']])

#transform Accident_Type_Code
x_train['Accident_Type_Code'] = x_train['Accident_Type_Code'].astype('category')
accident_type_code_dummies = pd.get_dummies(x_train.Accident_Type_Code,prefix="accident_type_code")
x_train = pd.concat([x_train, accident_type_code_dummies], axis=1)
x_train.drop(['Accident_Type_Code'],axis=1,inplace=True)

x_val['Accident_Type_Code'] = x_val['Accident_Type_Code'].astype('category')
val_accident_type_code_dummies = pd.get_dummies(x_val.Accident_Type_Code,prefix="accident_type_code")
x_val = pd.concat([x_val, val_accident_type_code_dummies], axis=1)
x_val.drop(['Accident_Type_Code'],axis=1,inplace=True)

df_test_x['Accident_Type_Code'] = df_test_x['Accident_Type_Code'].astype('category')
act_dummies = pd.get_dummies(df_test_x.Accident_Type_Code,prefix="accident_type_code")
df_test_x = pd.concat([df_test_x, act_dummies], axis=1)
df_test_x.drop(['Accident_Type_Code'],axis=1,inplace=True)

#Take violations as dummies
x_train['Violations'] = x_train['Violations'].astype('category')
Violations_dummies = pd.get_dummies(x_train.Violations,prefix="Violations")
x_train = pd.concat([x_train, Violations_dummies], axis=1)
x_train.drop(['Violations'],axis=1,inplace=True)

x_val['Violations'] = x_val['Violations'].astype('category')
Violations_dummies_val = pd.get_dummies(x_val.Violations,prefix="Violations")
x_val = pd.concat([x_val, Violations_dummies_val], axis=1)
x_val.drop(['Violations'],axis=1,inplace=True)

df_test_x['Violations'] = df_test_x['Violations'].astype('category')
Violations_dummies_test = pd.get_dummies(df_test_x.Violations,prefix="Violations")
df_test_x = pd.concat([df_test_x, Violations_dummies_test], axis=1)
df_test_x.drop(['Violations'],axis=1,inplace=True)


################################## xgBoost model  ########################################
xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
#xgb_model = xgb.XGBClassifier(objective="multi:softmax", random_state=42,num_class=4)
xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_train)


print(metrics.confusion_matrix(y_train, y_pred))
print(metrics.classification_report(y_train, y_pred))

#test xgb model on validation set
y_pred_val = xgb_model.predict(x_val)
print(metrics.classification_report(y_val, y_pred_val))


rfc_model =  RandomForestClassifier(n_estimators = 300, criterion = 'gini', random_state = 42)
rfc_model.fit(x_train, y_train)
y_pred_rfc = rfc_model.predict(x_train)
print(metrics.confusion_matrix(y_train, y_pred_rfc))
print(metrics.classification_report(y_train, y_pred_rfc))

y_pred_rfc_val = rfc_model.predict(x_val)
print(metrics.classification_report(y_val, y_pred_rfc_val))

### Random forest classifier is performing well

################### Prediction for test data final dataset ###################################

y_pred_results = rfc_model.predict(df_test_x)
df_test['Severity'] = y_pred_results

header = ["Accident_ID", "Severity"]
df_test.to_csv(r"E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\results-random-3.csv", columns = header,index=False)


y_pred_results_xgb = xgb_model.predict(df_test_x)
df_test['Severity'] = y_pred_results_xgb

header = ["Accident_ID", "Severity"]
df_test.to_csv(r"E:\MyDrive-2\DataScience\hacker-earth\he-airplane-accident\results-xgb.csv", columns = header,index=False)