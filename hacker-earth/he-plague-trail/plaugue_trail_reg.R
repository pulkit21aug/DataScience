library(dplyr)
library(tidyverse)
library(sentimentr)
library(modelr)
library(lubridate)
library(forecast)
library(pracma)
library(dynlm)


test <- read.csv("test.csv")
train <- read.csv("train.csv")


#plot - check for stationarity in our case it is not - hence we will use dynamic linear model
plot.ts(train$PA)
plot.ts(train$PB)
plot.ts(train$PC)
plot.ts(train$PD)
plot.ts(train$PE)
plot.ts(train$PF)
plot.ts(train$PG)

train_df <- train %>%select(PA,PB,PC,PD,PE,PF,PG)
cor(train_df)
#high correlation so one variable is sufficient to explain the model

str(train)

# fit model
model <- dynlm(log(PA)~TempOut+HeatIndex+OutHum+InTemp+InHum+log(PA),data=train)



summary(model)
anova(model)
plot(model)

test_data = add_predictions(test, model, var = "PA", type = NULL)

results_data <- test_data %>%
  select(ID,PA)
results_data$PA_unlog <- round(exp(results_data$PA))

write.csv(results_data,file = "MyData_2.csv")
