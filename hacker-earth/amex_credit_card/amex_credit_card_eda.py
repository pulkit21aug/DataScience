import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the train data set and test data set
df_train = pd.read_csv(r'D:\github\DataScience\hacker-earth\amex_credit_card\train.csv')
df_test = pd.read_csv(r'D:\github\DataScience\hacker-earth\amex_credit_card\test.csv')

################################# Exploratory Data analysis ###########################################
# name of the columns
df_train.columns

# Index(['customer_id', 'name', 'age', 'gender', 'owns_car', 'owns_house',
#        'no_of_children', 'net_yearly_income', 'no_of_days_employed',
#        'occupation_type', 'total_family_members', 'migrant_worker',
#        'yearly_debt_payments', 'credit_limit', 'credit_limit_used(%)',
#        'credit_score', 'prev_defaults', 'default_in_last_6months',
#        'credit_card_default'],
#       dtype='object')

df_train.rename(columns={'credit_limit_used(%)': 'credit_limit_used'}, inplace=True)
df_test.rename(columns={'credit_limit_used(%)': 'credit_limit_used'}, inplace=True)

# check for null values in any column
df_train.isnull().sum()  ## null value found

# customer_id                  0
# name                         0
# age                          0
# gender                       0
# owns_car                   547
# owns_house                   0
# no_of_children             774
# net_yearly_income            0
# no_of_days_employed        463
# occupation_type              0
# total_family_members        83
# migrant_worker              87
# yearly_debt_payments        95
# credit_limit                 0
# credit_limit_used(%)         0
# credit_score                 8
# prev_defaults                0
# default_in_last_6months      0
# credit_card_default          0

# check if data is balanced w.r.t to target variable

# check for balance of data - group by and check counts
df_target_var = df_train.groupby('credit_card_default').size().reset_index(name='counts')

# Create a pie chart
plt.clf()
plt.pie(
    # using data total)arrests
    df_target_var['counts'],
    # with the labels being officer names
    labels=df_target_var['credit_card_default'],
    # with no shadows
    shadow=False,
    # # with colors
    # colors=colors,
    # with one slide exploded out
    explode=(0, 0),
    # with the start angle at 90%
    startangle=90,
    # with the percent listed as a fraction
    autopct='%1.1f%%',
)

# View the plot drop above
plt.axis('equal')

# View the plot
plt.tight_layout()
plt.show()
###  Pie chart shows that data is highly imbalanced  with minority as defaulters

## Decision trees frequently perform well on imbalanced data -Use RandomForestClassifier
# EDA  - # age     0 - No missing values
df_train['age'].describe()

# count    45528.000000
# mean        38.993411
# std          9.543990
# min         23.000000
# 25%         31.000000
# 50%         39.000000
# 75%         47.000000
# max         55.000000
bins = [18, 24, 40, 56, 66]
labels = ['Z', 'Millennial', 'X', 'Boomers']
df_train['age_group'] = pd.cut(df_train['age'], bins=bins, labels=labels)
df_test['age_group'] = pd.cut(df_train['age'], bins=bins, labels=labels)

plt.clf()
plt.xscale('log')
bins = 1.15 ** (np.arange(0, 50))
plt.hist(df_train[df_train['credit_card_default'] == 1]['age_group'], bins=bins, alpha=0.8)
plt.hist(df_train[df_train['credit_card_default'] == 0]['age_group'], bins=bins, alpha=0.8)
# plt.legend(1, 0)
plt.show()
# Seems like Millenials have high case of defaulters
# Categorical variable - One Hot Encoding needed for  gender

#########EDA # gender  0  - No missing values
df_train['gender'].unique()
df_test['gender'].unique()

# Out[22]: array(['F', 'M', 'XNA'], dtype=object)
# Categorical variable - One Hot Encoding needed for  gender

##EDA # owns_car 547 records missing
df_train.count()
customer_id
45528
owns_car
44981
miss_car = (547 / 45528) * 100
# 1.20 % of the data are missing or null values -- this is inconsequential so we can easily drop records
# with missing value
df_train['owns_car'].describe()
df_train['owns_car'].unique()
## check test data if test dataset has missing values
df_test['owns_car'].describe()
df_test['owns_car'].unique()
# Out[40]: array(['Y', 'N', nan], dtype=object)
## since test data also has missing records we can not drop .
from statistics import mode

mode(df_train['owns_car'])
## 'N'
df_train['owns_car'].fillna('N', inplace=True)
df_test['owns_car'].fillna('N', inplace=True)

#########EDA # owns_house    0 - NO MISSING VALUES
df_train['owns_house'].describe()
df_train['owns_house'].unique()

plt.clf()
plt.xscale('log')
bins = 1.15 ** (np.arange(0, 50))
plt.hist(df_train[df_train['credit_card_default'] == 1]['owns_house'], bins=bins, alpha=0.8)
plt.hist(df_train[df_train['credit_card_default'] == 0]['owns_house'], bins=bins, alpha=0.8)
plt.legend('1', '0')
plt.show()
### mostly defaulters are those who dont have a house


#### EDA  no_of_children  774 missing values
df_train['no_of_children'].describe()
df_train['no_of_children'].mode()

### EDA net_yearly_income 0 - No missing  record
df_train['net_yearly_income'].describe()
# standard scalar usage

# EDA no_of_days_employed        463 - Missing records replace with mean
df_train['no_of_days_employed'].describe()
df_train['no_of_days_employed'].mean()

# EDA occupation_type  0 No missing records
df_train['occupation_type'].unique()
# candidate for Label Encoder with  categorical values

#EDA # total_family_members  83 missing values
df_train['total_family_members'].mode() #2

# EDA migrant_worker  87 missing  values
df_train['migrant_worker'].mode() ## fillna with 0
# since value are 0 and 1 no further processing

#EDA yearly_debt_payments    95
df_train['yearly_debt_payments'].mean() # fillna with mean value 31796

# credit_limit     0 No missing records however credit utilisation is a  better
#predictor as per domain so this can be dropped from the model

# credit_limit_used(%)   0 No missing required can be fed in model as it i
# since it is continuos variable
df_train.rename(columns={'credit_limit_used(%)': 'credit_limit_used'}, inplace=True)
df_test.rename(columns={'credit_limit_used(%)': 'credit_limit_used'}, inplace=True)

# EDA  credit_score  8
df_train['credit_score'].mean() #fil na woith 782

#EDA prev_defaults  0- can be fed as it is in model continuos variable
df_train['prev_defaults'].unique()

# EDA default_in_last_6months  0 - categorical variable with zero and one can be fed as it is
#in the model without any further processing