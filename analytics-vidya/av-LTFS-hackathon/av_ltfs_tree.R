

#install.packages("caret")
library(caret)
?train

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3333)
dtree_fit <- train(formula, data = train_subset_1, method = "manb",
                   parms = list(split = "information"),
                   trControl=trctrl,
                   tuneLength = 10)

train_data_tree = add_predictions(train_subset_1, dtree_fit, var = "loan_default", type = "raw")



#hist(train_data_tree$loan_default)

#train_data_treet$loan_default <- ifelse(train_data_tree$loan_default>=.34, 1, 0)
#xtab <- table(train_data_result$loan_default, train_subset_1$loan_default)
#confusionMatrix(xtab)



test_data_tree = add_predictions(test_subset_1, dtree_fit, var = "loan_default", type = "raw")
#test_data_tree$loan_default <- ifelse(test_data_tree$loan_default>=.34, 1, 0)


write.csv(test_data_tree,file = "tree_results.csv")


