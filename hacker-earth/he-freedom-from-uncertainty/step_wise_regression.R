library(tidyr)
library(dplyr)
library(MASS)
library(modelr)


train <- read.csv(file = "train.csv",header = TRUE)
test <- read.csv(file ="test.csv",header = TRUE)
test$Price <- 1

boxplot(train$ATR)

train.df <- na.omit(train)

model <- lm(Price ~. -ID -Date -Company, data = train.df)

summary(model)
anova(model)


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

final.sub <- data.frame(ID = test$ID ,Price = lm.prediction)

write.csv(test_data_result ,"results1.csv")

?randomForest
library( randomForest)

rf.model <-  randomForest(formula=formula_step,data = train.df)

test_data_result1 = add_predictions(test, rf.model, var = "Price", type = "response")

write.csv(test_data_result1 ,"randome_forest.csv")
