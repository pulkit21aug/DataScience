library(tidyr)
library(dplyr)
library(MASS)
library(modelr)


train <- read.csv(file = "train.csv",header = TRUE)
test <- read.csv(file ="test.csv",header = TRUE)

summary(train)
