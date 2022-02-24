library(tidyr)
library(dplyr)
library(modelr)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)


train <- read.csv(file = "train.csv",header = TRUE)
test <- read.csv(file ="test.csv",header = TRUE)



train.df <- na.omit(train)
train_xgb_df <-  train.df %>% dplyr::select(Price,SMA,EMA,WMA,DEMA,TEMA,TRIMA,KAMA,FAMA,T3,MACD_Hist,MACD_Signal,
                                         MAC_Hist,SlowK,FastK,ADXR,BOP,CMO,ULTOSC,MINUS_DI,PLUS_DI,MINUS_DM,
                                         PLUS_DM,MIDPOINT,SAR,TRANGE,
                                         ATR,ADOSC,OBV,HT_TRENDLINE,TRENDMODE,QUADRATURE )


test_xgb_df <-  test %>% dplyr::select(SMA,EMA,WMA,DEMA,TEMA,TRIMA,KAMA,FAMA,T3,MACD_Hist,MACD_Signal,
                                            MAC_Hist,SlowK,FastK,ADXR,BOP,CMO,ULTOSC,MINUS_DI,PLUS_DI,MINUS_DM,
                                            PLUS_DM,MIDPOINT,SAR,TRANGE,
                                            ATR,ADOSC,OBV,HT_TRENDLINE,TRENDMODE,QUADRATURE )



train_xgb_df_matrix <- as.matrix(train_xgb_df)
test_xgb_df_matrix <- as.matrix(test_xgb_df)



xgb_model <- xgboost(train_xgb_df_matrix[,-1], 
               label = train_xgb_df_matrix[,1], 
               eta = 0.2,
               max_depth = 15, 
               nround=400, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               objective = "reg:linear",
               nthread = 3
)

summary(xgb_model)
print(xgb_model$evaluation_log)



y_pred <- predict(xgb_model, test_xgb_df_matrix)

test_data_result <- cbind(test,y_pred)



write.csv(test_data_result ,"xgboost8.csv")
