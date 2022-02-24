import keras  as keras
import tensorflow as tensorflow
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import  confusion_matrix
from keras.callbacks import EarlyStopping


train  = pd.read_csv(r"E:\MyDrive-2\DataScience\he-freedom-from-uncertainty\train.csv")
test  = pd.read_csv(r"E:\MyDrive-2\DataScience\he-freedom-from-uncertainty\test.csv")
test_results = test

#train.hist(figsize=(69,10))

corr_matrix = train.corr(method='pearson').abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
to_drop.remove('Price')


train.drop(train[to_drop],axis=1,inplace=True)
test.drop(test[to_drop],axis=1,inplace=True)

train.columns
train = train.drop('ID',axis=1)
train = train.drop('Date',axis=1)
train = train.drop('Company ',axis=1)

test = test.drop('ID',axis=1)
test = test.drop('Date',axis=1)
test = test.drop('Company ',axis=1)

#train.hist(figsize = (40,10))


summary = train.describe()
summary = summary.transpose()
print(summary)
summary.head()

train.size

train = train.dropna(axis=0)
#train.boxplot('ATR')

summary = train.describe()
summary = summary.transpose()


#train.boxplot('Chaikin A/D')



def normalize(df):
    feature_names = ['Chaikin A/D', 'ADOSC', 'OBV']
    result = df.copy()
    for feature_name in feature_names:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

train_scaled = normalize(train)
test_scaled = normalize(test)

summary_train_scaled = train_scaled.describe()
summary_train_scaled = summary_train_scaled.transpose()

train_scaled_x = train_scaled.drop(columns=['Price'])
train_scaled_y = train_scaled[['Price']]

#create model
model = Sequential()

#get number of columns in training data
n_cols = train_scaled_x.shape[1]

#add model layers
model.add(Dense(350, activation='relu', input_shape=(n_cols,)))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=["mean_absolute_error"])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=20)
#train model
model.fit(train_scaled_x, train_scaled_y, batch_size=125, epochs=30, callbacks=[early_stopping_monitor])

results = model.evaluate(train_scaled_x, train_scaled_y)
#Print the results
for i in range(len(model.metrics_names)):
    print("Metric ",model.metrics_names[i],":",str(round(results[i],2)))

test_scaled_x_predictions = model.predict(test_scaled)
test_results['Price'] = test_scaled_x_predictions

header = ["ID", "Price"]
test_results.to_csv(r"E:\MyDrive-2\DataScience\he-freedom-from-uncertainty\output-dnn4.csv", columns = header)

train_scaled.to_csv(r"E:\MyDrive-2\DataScience\he-freedom-from-uncertainty\train_scaled.csv")
test_scaled.to_csv(r"E:\MyDrive-2\DataScience\he-freedom-from-uncertainty\test_scaled.csv")

train_scaled_x