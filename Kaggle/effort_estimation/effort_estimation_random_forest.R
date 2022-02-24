library(modelr)
library (ridge)
library(ggplot2)
library(randomForest)
library(caret)
library(e1071)


train <- read.csv("train_1.csv",sep = ",")

# model <- lm(Actual ~ getAll +  getByRef + create + multi_create + update +
#               multi_update + delete + multi_delete+multi_upsert + Complexity,data = train)
# 
# summary(model)
# 
# anova(model)


# Define the control
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid")

set.seed(1234)
# Run the model


model <- train(Actual ~ getAll +  getByRef + create + multi_create + update + multi_update + delete + multi_delete+multi_upsert + Complexity,
                    data = train,
                    method = "rf",
                    metric = "mse",
                    trControl = trControl)
# Print the results
print(model)



results <- add_predictions(train, model, var = "pred_estimation", type = NULL)

results <- results[order(results$Actual),]


#Plot Actual vs Predicted values for Test Cases                  
ggplot(results,aes(x =1:nrow(train),color=Series)) +
  geom_line(data = results, aes(x =1:nrow(train), y = Actual, color ="Actual")) +
  geom_line(data = results, aes(x =1:nrow(train), y = pred_estimation, color ="Predicted"))  +xlab('Estimates') +ylab('Trend Line')


test <- read.csv("test_1.csv",sep = ",")
test_results <- add_predictions(test, model, var = "pred_estimation", type = NULL)
