library(tidyr)
library(dplyr)


train <- read.csv(file = "train.csv",header = TRUE)
test <- read.csv(file ="test.csv",header = TRUE)
test$Price <- 1


train.df <- na.omit(train)
# model <- lm(Price ~. -ID -Date -Company, data = train.df)
# summary(model)
# anova(model)



train.df.pca <- train.df %>% dplyr::select(Price,SMA,EMA,WMA,DEMA,TEMA,TRIMA,KAMA,FAMA,T3,MACD_Hist,MACD_Signal,
                                    MAC_Hist,SlowK,FastK,ADXR,BOP,CMO,ULTOSC,MINUS_DI,PLUS_DI,MINUS_DM,
                                    PLUS_DM,MIDPOINT,SAR,TRANGE,
                                    ATR,ADOSC,OBV,HT_TRENDLINE,TRENDMODE,QUADRATURE)

test.df.pca <- test %>% dplyr::select(Price,SMA,EMA,WMA,DEMA,TEMA,TRIMA,KAMA,FAMA,T3,MACD_Hist,MACD_Signal,
                               MAC_Hist,SlowK,FastK,ADXR,BOP,CMO,ULTOSC,MINUS_DI,PLUS_DI,MINUS_DM,
                               PLUS_DM,MIDPOINT,SAR,TRANGE,
                               ATR,ADOSC,OBV,HT_TRENDLINE,TRENDMODE,QUADRATURE)


res <- cor(train.df.pca ,use = "complete.obs")
round(res, 2)



train_pca <- prcomp(train.df.pca[,c(8:32)], center = TRUE,scale. = TRUE)
summary(train_pca)
train_pca$center

train.data <- data.frame(Price = train.df.pca$Price, train_pca$x)

train.data <- train.data[,1:4]
lm_model <- lm(log(Price) ~. ,data = train.data)
summary(lm_model)

test.data <- predict(train_pca, newdata = test.df.pca)
test.data <- as.data.frame(test.data)
lm.prediction <- exp(predict(lm_model, test.data))

final.sub <- data.frame(ID = test$ID ,Price = lm.prediction)

write.csv(final.sub,"pca2.csv")

