# transform train/test data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler


def eda(df):
    # Exploratory data analysis
    # Target variable analysis
    plt.hist(df['Survived'])
    plt.show()
    # check for unique values - null or nan can be checked since categorical
    df['Survived'].value_counts(dropna=False)
    # Out[11]:
    # 0    549
    # 1    342
    # Name: Survived, dtype: int64

    # check for null columns
    null_columns = df.columns[df.isnull().any()]
    df[null_columns].isnull().sum()
    # Age         177
    # Cabin       687

    df['Pclass'].unique()  # Out[10]: array([3, 1, 2], dtype=int64)
    df['Pclass'].value_counts(dropna=False)
    # Out[13]:
    # 3    491
    # 1    216
    # 2    184

    df['Sex'].unique()
    # Out[14]: array(['male', 'female'], dtype=object)
    df['Sex'].value_counts(dropna=False)
    # male      577
    # female    314
    # Name: Sex, dtype: int64
    plt.hist(df['Sex'])
    plt.show()

    df['SibSp'].unique()
    # Out[6]: array([1, 0, 3, 4, 2, 5, 8], dtype=int64)
    df['SibSp'].value_counts(dropna=False)
    # Out[7]:
    # 0    608
    # 1    209
    # 2     28
    # 4     18
    # 3     16
    # 8      7
    # 5      5
    df['Parch'].unique()
    # Out[8]: array([0, 1, 2, 5, 3, 4, 6], dtype=int64)
    df['Parch'].value_counts(dropna=False)
    # 0    678
    # 1    118
    # 2     80
    # 5      5
    # 3      5
    # 4      4
    # 6      1

    # transform age into bin
    plt.hist(df['Age'])
    plt.show()
    df['Age'].max()
    df['Age'].describe()  # mean age is 29 - mean replacement for NAN
    df['age_bins'].unique()
    df['age_bins'].value_counts(dropna=False)

    # transform fair into bin
    plt.hist(df['Fare'])
    plt.show()
    df['Fare'].describe()
    df['Embarked'].value_counts(dropna=False)



def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

def transform_data(df):
    # create dummy variables for Pclass
    df_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass', drop_first=True)
    # Join the dummy variables to the main dataframe
    df = pd.concat([df, df_Pclass], axis=1)
    del df['Pclass']

    # transform column sex
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])

    # transform age - mean replacement

    mean_age = df['Age'].mean()
    df['Age'] = df['Age'].apply(lambda x: mean_age if pd.isnull(x) else x)
    # create buckets or bins to categorize passengers agewise
    cut_bins = [0, 20, 40, 80]
    cut_labels = ["young", "working", "old"]
    df['age_bins'] = pd.cut(df['Age'], bins=cut_bins, labels=cut_labels)
    # Nan handling
    df['age_bins'] = df['age_bins'].fillna("working")
    del df['Age']

    # create dummies for age_bins
    df_age_bins = pd.get_dummies(df['age_bins'], prefix='age_bins', drop_first=True)
    df = pd.concat([df, df_age_bins], axis=1)
    del df['age_bins']

    f_bins = [0, 50, 100, 600]
    f_labels = ["low", "mid", "high"]
    df['Fare_bins'] = pd.cut(df['Fare'], bins=f_bins, labels=f_labels)
    del df['Fare']
    # create dummies for Fare
    df_Fare_bins = pd.get_dummies(df['Fare_bins'], prefix='Fare_bins', drop_first=True)
    df = pd.concat([df, df_Fare_bins], axis=1)
    del df['Fare_bins']

    # handle cabin
    # df['Cabin'] = df['Cabin'].apply(lambda x: 'NO_CABIN' if pd.isnull(x) else 'CABIN')
    # df['Cabin'] = label_encoder.fit_transform(df['Cabin'])
    del df['Cabin']

    # clean Embarked
    df['Embarked'].value_counts(dropna=False)
    df['Embarked'] = df['Embarked'].apply(lambda x: 'S' if pd.isnull(x) else x)

    # create dummies for Embarked
    df_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
    df = pd.concat([df, df_Embarked], axis=1)
    del df['Embarked']

    # delete not used columns
    del df['Name']
    del df['Ticket']
    del df['PassengerId']
    if 'Survived' in df:
        del df['Survived']
    return df
