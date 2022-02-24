library(tidyr)
library(dplyr)
library(MASS)
library(modelr)


train <- read.csv(file = "train_scaled.csv",header = TRUE)
test <- read.csv(file ="test_scaled.csv",header = TRUE)
test$Price <- 1

train.df <- na.omit(train)

model <- lm(Price ~. -ID -Date -Company, data = train.df)
# Stepwise regression model
step.model <- stepAIC(model , direction = "both", 
                      trace = FALSE)
summary(step.model)

train.df.pca <- train.df %>% select(SMA,EMA,WMA,DEMA,TEMA,TRIMA,KAMA,FAMA,MAMA,T3,MACD,
                                    MAC,MAC_Hist,MAC_Signal,SlowD,SlowK,FastK,RSI,FatD,
                                    FatK,WILLR,APO,MOM,BOP,CCI,ULTOSC,MINUS_DI,PLUS_DI,
                                    MINUS_DM,PLUS_DM,MIDPOINT,SAR,TRANGE,ATR,ADOSC,OBV,HT_TRENDLINE,Price)

formula_step = Price ~ SMA + EMA + WMA + DEMA + TEMA + TRIMA + 
  KAMA + FAMA + T3 + MACD_Hist + MACD_Signal + MAC_Hist + SlowK + 
  FastK + ADXR + BOP + CMO + ULTOSC + MINUS_DI + PLUS_DI + 
  MINUS_DM + PLUS_DM + MIDPOINT + SAR + TRANGE + ATR + ADOSC + 
  OBV + HT_TRENDLINE + TRENDMODE + QUADRATURE

test_data_result = add_predictions(test, step.model, var = "Price", type = "response")

final.sub <- data.frame(ID = test$ID ,Price = test_data_result$Price)

write.csv(test_data_result ,"step-scaled.csv")

?randomForest
library( randomForest)

rf.model <-  randomForest(formula=formula_step,data = train.df)

test_data_result1 = add_predictions(test, rf.model, var = "Price", type = "response")

write.csv(test_data_result1 ,"randome_forest.csv")
