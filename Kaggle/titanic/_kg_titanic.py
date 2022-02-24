# Classification Problem - Titanic Survival
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from com.puls.kaggle.titanic._transform import transform_data, Find_Optimal_Cutoff
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv(r"E:\MyDrive-2\DataScience\Kaggle\titanic\train.csv")
test = pd.read_csv(r"E:\MyDrive-2\DataScience\Kaggle\titanic\test.csv")

plt.hist(train['Age'])
plt.show()

#data cleaning and transformation
train_x =  transform_data(train)
test_x =    transform_data(test)
train_y = train['Survived']

#xgb model
model = XGBClassifier(n_estimators=100 ,max_depth=9, seed=2017)
# model = LogisticRegression()
# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=10, random_state=2017)
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
print ("AUC :", metrics.roc_auc_score(train_y, y_pred_prob))

# extract false positive, true positive rate
fpr, tpr, thresholds = metrics.roc_curve(train_y, y_pred_prob)
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i),'1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i),'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'], label='tpr')
plt.plot(roc['1-fpr'], color = 'red', label='1-fpr')
plt.legend(loc='best')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()
#----output----

# Find optimal probability threshold
# Note: probs[:, 1] will have probability of being positive label
threshold = Find_Optimal_Cutoff(train_y, probs[:, 1])
print("Optimal Probability Threshold: ", threshold)

# Applying the threshold to the prediction probability
y_pred_optimal = np.where(y_pred_prob >= threshold, 1, 0)

# Let's compare the accuracy of traditional/normal approach vs optimal cutoff
print ("\nNormal - Accuracy: ", metrics.accuracy_score(train_y, y_pred))
print ("Optimal Cutoff - Accuracy: ", metrics.accuracy_score(train_y, y_pred_optimal))
print ("\nNormal - Confusion Matrix: \n", metrics.confusion_matrix(train_y, y_pred))
print ("Optimal - Cutoff Confusion Matrix: \n", metrics.confusion_matrix(train_y, y_pred_optimal))



feature_importance = model.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, train_x.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

#Test data set Prediction
test_pred = model.predict(test_x)
test['Survived'] = test_pred
header = ["PassengerId","Survived"]
test.to_csv(r"E:\MyDrive-2\DataScience\Kaggle\titanic\xgboost.csv", columns=header ,index=False)



