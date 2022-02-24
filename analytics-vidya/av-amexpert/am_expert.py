import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

# read data sets
train = pd.read_csv(r"E:\MyDrive-2\DataScience\av-amexpert\train.csv")
test = pd.read_csv(r"E:\MyDrive-2\DataScience\av-amexpert\test.csv")
campaign_data = pd.read_csv(r"E:\MyDrive-2\DataScience\av-amexpert\campaign_data.csv")
coupon_item_mapping = pd.read_csv(r"E:\MyDrive-2\DataScience\av-amexpert\coupon_item_mapping.csv")
customer_demographics = pd.read_csv(r"E:\MyDrive-2\DataScience\av-amexpert\customer_demographics.csv")
customer_transaction_data = pd.read_csv(r"E:\MyDrive-2\DataScience\av-amexpert\customer_transaction_data.csv")
item_data = pd.read_csv(r"E:\MyDrive-2\DataScience\av-amexpert\item_data.csv")
customer_transaction_data['sp'] = customer_transaction_data['selling_price']/customer_transaction_data['quantity']
customer_transaction_data = customer_transaction_data.drop_duplicates()

train.shape

mydict = dict(zip(coupon_item_mapping.coupon_id, coupon_item_mapping.item_id))
train = pd.merge(train, campaign_data, on="campaign_id", how="left")
train = pd.merge(train, customer_demographics, on="customer_id", how="left")
train['item_id'] = train['coupon_id'].map(mydict)
train = pd.merge(train, item_data, on="item_id", how="left")
ctx_dict = customer_transaction_data.set_index(['customer_id','item_id']).to_dict('records')

train['sp'] = train['customer_id','item_id'].map(ctx_dict)['sp']


for col in train.columns:
    print(col)

summary = train.describe()
summary = summary.transpose()

train_final = train.drop(columns=['start_date', 'end_date', 'brand'])

# missing nan handling for every column
# train_final.isna().sum()
# id                         0
# campaign_id                0
# coupon_id                  0
# customer_id                0
# redemption_status          0
# campaign_type              0
# item_id                    0
# brand_type                 0
# category                   0
# age_range            4200942
# marital_status       4200942
# rented               2612838
# family_size          2612838
# no_of_children       5263222
# income_bracket       2612838

# data cleaning
train_final['age_range'].unique()
train_final['age_range'] = train_final['age_range'].fillna('age_na')

train_final['marital_status'].unique()
train_final['marital_status'] = train_final['marital_status'].fillna('ms_na')

train_final['rented'].unique()
# replace with 2 for nan
train_final['rented'] = train_final['rented'].fillna(2)

train_final['family_size'].unique()
train_final['family_size'] = train_final['family_size'].fillna('fs_na')

train_final['no_of_children'].unique()
train_final['no_of_children'] = train_final['no_of_children'].fillna('noc_na')

train_final['income_bracket'].unique()
# mean replacement
train_final['income_bracket'] = train_final['income_bracket'].fillna(4.79)

train_final['campaign_type'].unique()
train_final['brand_type'].unique()
train_final['category'].unique()

# model implementation - start
y = train_final.redemption_status
col = "campaign_type brand_type category age_range marital_status rented family_size no_of_children income_bracket sp".split()
x = pd.DataFrame(train_final, columns=col)

labelencoder = LabelEncoder()
x['age_range'] = labelencoder.fit_transform(x['age_range'])
x['marital_status'] = labelencoder.fit_transform(x['marital_status'])
x['family_size'] = labelencoder.fit_transform(x['family_size'])
x['no_of_children'] = labelencoder.fit_transform(x['no_of_children'])
x['campaign_type'] = labelencoder.fit_transform(x['campaign_type'])
x['brand_type'] = labelencoder.fit_transform(x['brand_type'])
x['category'] = labelencoder.fit_transform(x['category'])

model = XGBClassifier()
model.fit(x, y)
y_pred = model.predict(x)
cm = confusion_matrix(y, y_pred)
print(cm)
# model implementation - end

# test data preparation
test.shape
test = pd.merge(test, campaign_data, on="campaign_id", how="left")
test = pd.merge(test, customer_demographics, on="customer_id", how="left")
test['item_id'] = test['coupon_id'].map(mydict)
test = pd.merge(test, item_data, on="item_id", how="left")
test['sp'] = test['item_id'].map(ctx_dict)

test_final = test.drop(columns=['start_date', 'end_date', 'brand'])
test_final['age_range'] = test_final['age_range'].fillna('age_na')
test_final['marital_status'] = test_final['marital_status'].fillna('ms_na')
test_final['rented'] = test_final['rented'].fillna(2)
test_final['family_size'] = test_final['family_size'].fillna('fs_na')
test_final['no_of_children'] = test_final['no_of_children'].fillna('noc_na')
test_final['income_bracket'] = test_final['income_bracket'].fillna(4.79)

test_x = pd.DataFrame(test_final, columns=col)

test_x['age_range'] = labelencoder.fit_transform(test_x['age_range'])
test_x['marital_status'] = labelencoder.fit_transform(test_x['marital_status'])
test_x['family_size'] = labelencoder.fit_transform(test_x['family_size'])
test_x['no_of_children'] = labelencoder.fit_transform(test_x['no_of_children'])
test_x['campaign_type'] = labelencoder.fit_transform(test_x['campaign_type'])
test_x['brand_type'] = labelencoder.fit_transform(test_x['brand_type'])
test_x['category'] = labelencoder.fit_transform(test_x['category'])

test_y_pred = model.predict(test_x)


test['redemption_status'] = test_y_pred

header = ["id","redemption_status"]
test.to_csv(r"E:\MyDrive-2\DataScience\av-amexpert\output-xgboost-6.csv", columns=header ,index=False)
