library(tidyr)
library(dplyr)
library(MASS)
library(modelr)
library( randomForest)


train <- read.csv(file = "train.csv",header = TRUE)
test <- read.csv(file ="test.csv",header = TRUE)
test$Price <- 1


train.df <- na.omit(train)


formula_step = log(Price) ~ SMA + EMA + WMA + DEMA + TEMA + TRIMA + 
  KAMA + FAMA + T3 + MACD_Hist + MACD_Signal + MAC_Hist + SlowK + 
  FastK + ADXR + BOP + CMO + ULTOSC + MINUS_DI + PLUS_DI + 
  MINUS_DM + PLUS_DM + MIDPOINT + SAR + TRANGE + ATR + ADOSC + 
  OBV + HT_TRENDLINE + TRENDMODE + QUADRATURE

reg_model <- lm(formula_step,data=train)

summary(reg_model)
?randomForest

rf.model <-  randomForest(formula=formula_step,data = train.df)
summary(rf.model$mse)

test_data_result1 = add_predictions(test, rf.model, var = "Price", type = "response")

write.csv(test_data_result1 ,"randome_forest.csv")


test_data_result2 = add_predictions(test, reg_model, var = "Price", type = "response")

write.csv(test_data_result2 ,"linear_reg.csv")
