library(dplyr)
library(tidyr)
library(modelr)
library(caret)
#install.packages("ROCR")
library(ROCR)

train <- read.csv("train.csv")

colnames(train)

train_subset_1 <- train %>% select(UniqueID,disbursed_amount,asset_cost,ltv,Date.of.Birth,Employment.Type,
                           DisbursalDate,MobileNo_Avl_Flag,Aadhar_flag,PAN_flag,VoterID_flag,
                           Driving_flag,Passport_flag,PERFORM_CNS.SCORE,PERFORM_CNS.SCORE.DESCRIPTION,PRI.NO.OF.ACCTS,
                           PRI.ACTIVE.ACCTS,PRI.OVERDUE.ACCTS,PRI.CURRENT.BALANCE,SEC.NO.OF.ACCTS,SEC.ACTIVE.ACCTS,SEC.OVERDUE.ACCTS,SEC.CURRENT.BALANCE,
                           AVERAGE.ACCT.AGE,CREDIT.HISTORY.LENGTH,NO.OF_INQUIRIES,NEW.ACCTS.IN.LAST.SIX.MONTHS,loan_default)

colnames(train_subset_1)

#convert loan_default
train_subset_1$loan_default <- as.factor(train_subset_1$loan_default)

#convert dob availability
str(train_subset_1$Date.of.Birth)
train_subset_1$Date.of.Birth <-as.character(train_subset_1$Date.of.Birth)
train_subset_1$Date.of.Birth <- as.factor(ifelse(train_subset_1$Date.of.Birth=="01-01-00", 0, 1))

#convert primary and secondary account balance since negative may be account closed
train_subset_1$PRI.CURRENT.BALANCE <- ifelse(train_subset_1$PRI.CURRENT.BALANCE<=0 ,0 ,train_subset_1$PRI.CURRENT.BALANCE)
train_subset_1$SEC.CURRENT.BALANCE <- ifelse(train_subset_1$SEC.CURRENT.BALANCE<=0 ,0 ,train_subset_1$SEC.CURRENT.BALANCE)


#convert perform cns score
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- as.character(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
unique(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "C-Very Low Risk",1,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "A-Very Low Risk",1,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "D-Very Low Risk",1,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "B-Very Low Risk",1,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)

train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "G-Low Risk",2,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "E-Low Risk",2,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "F-Low Risk",2,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: Only a Guarantor",2,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)


train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "I-Medium Risk",3,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "H-Medium Risk",3,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "No Bureau History Available",3,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: Not Enough Info available on the customer",3,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: Sufficient History Not Available",3,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: No Activity seen on the customer (Inactive)",3,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: No Updates available in last 36 months",3,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)


train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "J-High Risk",4,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "K-High Risk",4,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)

train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "L-Very High Risk",5,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "M-Very High Risk",5,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)

train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: More than 50 active Accounts found",5,
                                                       train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- as.numeric(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)

str(train_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)

#convert employment type
train_subset_1$Employment.Type <- as.character(train_subset_1$Employment.Type)
train_subset_1$Employment.Type[train_subset_1$Employment.Type ==""] <-'NoInfo'
train_subset_1$Employment.Type <- as.factor(train_subset_1$Employment.Type)
train_subset_1$Employment.Type <- as.numeric(train_subset_1$Employment.Type)
train_subset_1$Employment.Type <- as.factor(train_subset_1$Employment.Type)



#current libailities
train_subset_1$curr_liab <- abs(train_subset_1$PRI.CURRENT.BALANCE)+abs(train_subset_1$SEC.CURRENT.BALANCE)

#current liabilities to future debt ratio
train_subset_1$clfd <- train_subset_1$curr_liab/train_subset_1$disbursed_amount 


#convert to factor -KYC 
train_subset_1$MobileNo_Avl_Flag <- as.factor(train_subset_1$MobileNo_Avl_Flag)
train_subset_1$Aadhar_flag <- as.factor(train_subset_1$Aadhar_flag)
train_subset_1$PAN_flag <- as.factor(train_subset_1$PAN_flag)
train_subset_1$VoterID_flag <- as.factor(train_subset_1$VoterID_flag)
train_subset_1$Driving_flag <- as.factor(train_subset_1$Driving_flag)
train_subset_1$Passport_flag <- as.factor(train_subset_1$Passport_flag)




#model 
formula <- loan_default ~ ltv + Date.of.Birth + Employment.Type+ 
  Aadhar_flag+ PERFORM_CNS.SCORE+PERFORM_CNS.SCORE.DESCRIPTION+
  +PRI.ACTIVE.ACCTS+PRI.OVERDUE.ACCTS+ curr_liab + disbursed_amount


str(train_subset_1)
summary(train_subset_1)

logModel <-  glm(formula,data=train_subset_1 ,family="binomial")
summary(logModel)
anova(logModel,test = "Chisq")

train_data_result = add_predictions(train_subset_1, logModel, var = "loan_default", type = "response")
ROCRPred <- prediction(train_data_result$loan_default,train_subset_1$loan_default)
ROCRPref <- performance(ROCRPred,"tpr","fpr")
plot(ROCRPref,colorize=TRUE,cutoffs ,at=seq(.1,by=.1))
acc.perf = performance(ROCRPred, measure = "acc")
plot(acc.perf)

hist(train_data_result$loan_default)

train_data_result$loan_default <- ifelse(train_data_result$loan_default>=.20, 1, 0)

xtab <- table(ActualValue=train_subset_1$loan_default,PredictedValue=train_data_result$loan_default )
confusionMatrix(xtab)


##Test model preparations
test <- read.csv("test.csv")

test_subset_1 <- test %>% select(UniqueID,disbursed_amount,asset_cost,ltv,Date.of.Birth,Employment.Type,
                                   DisbursalDate,MobileNo_Avl_Flag,Aadhar_flag,PAN_flag,VoterID_flag,
                                   Driving_flag,Passport_flag,PERFORM_CNS.SCORE,PERFORM_CNS.SCORE.DESCRIPTION,PRI.NO.OF.ACCTS,
                                   PRI.ACTIVE.ACCTS,PRI.OVERDUE.ACCTS,PRI.CURRENT.BALANCE,SEC.NO.OF.ACCTS,SEC.ACTIVE.ACCTS,SEC.OVERDUE.ACCTS,SEC.CURRENT.BALANCE,
                                   AVERAGE.ACCT.AGE,CREDIT.HISTORY.LENGTH,NO.OF_INQUIRIES,NEW.ACCTS.IN.LAST.SIX.MONTHS)


#convert dob availability
str(test_subset_1$Date.of.Birth)
test_subset_1$Date.of.Birth <-as.character(test_subset_1$Date.of.Birth)
test_subset_1$Date.of.Birth <- as.factor(ifelse(test_subset_1$Date.of.Birth=="01-01-00", 0, 1))

#convert primary and secondary account balance since negative may be account closed
test_subset_1$PRI.CURRENT.BALANCE <- ifelse(test_subset_1$PRI.CURRENT.BALANCE<=0 ,0 ,test_subset_1$PRI.CURRENT.BALANCE)
test_subset_1$SEC.CURRENT.BALANCE <- ifelse(test_subset_1$SEC.CURRENT.BALANCE<=0 ,0 ,test_subset_1$SEC.CURRENT.BALANCE)


#convert perform cns score
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- as.character(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
unique(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "C-Very Low Risk",1,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "A-Very Low Risk",1,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "D-Very Low Risk",1,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "B-Very Low Risk",1,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)

test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "G-Low Risk",2,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "E-Low Risk",2,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "F-Low Risk",2,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: Only a Guarantor",2,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)


test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "I-Medium Risk",3,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "H-Medium Risk",3,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "No Bureau History Available",3,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: Not Enough Info available on the customer",3,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: Sufficient History Not Available",3,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: No Activity seen on the customer (Inactive)",3,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: No Updates available in last 36 months",3,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)


test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "J-High Risk",4,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "K-High Risk",4,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)

test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "L-Very High Risk",5,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "M-Very High Risk",5,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)

test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- ifelse(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION == "Not Scored: More than 50 active Accounts found",5,
                                                      test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)
test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION <- as.numeric(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)

str(test_subset_1$PERFORM_CNS.SCORE.DESCRIPTION)

#convert employment type
test_subset_1$Employment.Type <- as.character(test_subset_1$Employment.Type)
test_subset_1$Employment.Type[test_subset_1$Employment.Type ==""] <-'NoInfo'
test_subset_1$Employment.Type <- as.factor(test_subset_1$Employment.Type)
test_subset_1$Employment.Type <- as.numeric(test_subset_1$Employment.Type)
test_subset_1$Employment.Type <- as.factor(test_subset_1$Employment.Type)



#current libailities
test_subset_1$curr_liab <- abs(test_subset_1$PRI.CURRENT.BALANCE)+abs(test_subset_1$SEC.CURRENT.BALANCE)

#current liabilities to future debt ratio
test_subset_1$clfd <- test_subset_1$curr_liab/test_subset_1$disbursed_amount 


#convert to factor -KYC 
test_subset_1$MobileNo_Avl_Flag <- as.factor(test_subset_1$MobileNo_Avl_Flag)
test_subset_1$Aadhar_flag <- as.factor(test_subset_1$Aadhar_flag)
test_subset_1$PAN_flag <- as.factor(test_subset_1$PAN_flag)
test_subset_1$VoterID_flag <- as.factor(test_subset_1$VoterID_flag)
test_subset_1$Driving_flag <- as.factor(test_subset_1$Driving_flag)
test_subset_1$Passport_flag <- as.factor(test_subset_1$Passport_flag)


test_data_result = add_predictions(test_subset_1, logModel, var = "loan_default", type = "response")
test_data_result$loan_default <- ifelse(test_data_result$loan_default>=.20, 1, 0)
write.csv(test_data_result,file = "results_3.csv")
table(test_data_result$loan_default)

