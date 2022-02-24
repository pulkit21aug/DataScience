import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# dummy candidate gender ,owns_car ,owns_house


def age_to_range(df):
    bins = [18, 24, 40, 56, 66]
    labels = ['Z', 'Millennial', 'X', 'Boomers']
    df['age_range'] = pd.cut(df['age'], bins=bins, labels=labels)
    return df


def fill_na_owns_car(df):
    df['owns_car'].fillna('N', inplace=True)
    return df


def fill_na_no_of_children(df):
    df['no_of_children'].fillna(0, inplace=True)
    return df


def fill_na_no_of_days_employed(df):
    df['no_of_days_employed'].fillna(67609, inplace=True)
    return df


def fill_na_total_family_members(df):
    df['total_family_members'].fillna(2, inplace=True)
    return df


def fill_na_migrant_worker(df):
    df['migrant_worker'].fillna(0, inplace=True)
    return df


def fill_na_yearly_debt_payments(df):
    df['yearly_debt_payments'].fillna(31796, inplace=True)
    return df


def fill_na_credit_score(df):
    df['credit_score'].fillna(782, inplace=True)
    return df


def sd_net_yearly_income(df):
    sc = MinMaxScaler()
    df['net_yearly_income'] = sc.fit_transform(df[['net_yearly_income']])
    return df


def sd_yearly_debt_payments(df):
    sd = MinMaxScaler()
    df['yearly_debt_payments'] = sd.fit_transform(df[['yearly_debt_payments']])
    return df


def sd_no_of_days_employed(df):
    sc = MinMaxScaler()
    df['no_of_days_employed'] = sc.fit_transform(df[['no_of_days_employed']])
    return df


def sd_credit_score(df):
    sc = MinMaxScaler()
    df['credit_score'] = sc.fit_transform(df[['credit_score']])
    return df


def lb_encode_age_group(df):
    le = LabelEncoder()
    df['age_range'] = le.fit_transform(df['age_range'])
    return df


def lb_encode_occupation_type(df):
    lc = LabelEncoder()
    df['occupation_type'] = lc.fit_transform(df['occupation_type'])
    return df


def main():
    print("Utility methods for amex credit card dataset analysis")


if __name__ == "__main__":
    main()
