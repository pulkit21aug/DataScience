library(dplyr)
library(sentimentr)
library(modelr)

train_data <- read.csv("train.csv")

str(train_data)


#clean data - commment
# Then remove all the punctuation
train_data$comment_cleaned = gsub("[[:punct:]]", " ", train_data$comment)
# Then remove numbers, we need only text for analytics
train_data$comment_cleaned = gsub("[[:digit:]]", " ", train_data$comment_cleaned)
# finally, we remove unnecessary spaces (white spaces, tabs etc)
train_data$comment_cleaned = gsub("[ \t]{2,}", " ", train_data$comment_cleaned)
train_data$comment_cleaned = gsub("^\\s+|\\s+$", "", train_data$comment_cleaned)


#clean data - parent commment
# Then remove all the punctuation
train_data$pa_comment_cleaned = gsub("[[:punct:]]", " ", train_data$parent_comment)
# Then remove numbers, we need only text for analytics
train_data$pa_comment_cleaned = gsub("[[:digit:]]", " ", train_data$pa_comment_cleaned)
# finally, we remove unnecessary spaces (white spaces, tabs etc)
train_data$pa_comment_cleaned = gsub("[ \t]{2,}", " ", train_data$pa_comment_cleaned)
train_data$pa_comment_cleaned = gsub("^\\s+|\\s+$", "", train_data$pa_comment_cleaned)

#calculate comment_score
train_data$comment_score =  sentiment(get_sentences(train_data$comment_cleaned))$sentiment
train_data$pa_comment_score = sentiment(get_sentences(train_data$pa_comment_cleaned))$sentiment

summary(train_data)
colnames(train_data)

train_data <- train_data %>%
               select(UID,comment_score,pa_comment_score,score)


model <- lm(score ~  comment_score,data = train_data)

predict(model,train_data)

summary(model)

test_data <- read.csv("test.csv")

#clean data - parent commment
# Then remove all the punctuation
test_data$comment_cleaned = gsub("[[:punct:]]", " ", test_data$comment)
# Then remove numbers, we need only text for analytics
test_data$comment_cleaned = gsub("[[:digit:]]", " ", test_data$comment_cleaned)
# finally, we remove unnecessary spaces (white spaces, tabs etc)
test_data$comment_cleaned = gsub("[ \t]{2,}", " ", test_data$comment_cleaned)
test_data$comment_cleaned = gsub("^\\s+|\\s+$", "", test_data$comment_cleaned)

#calculate score
test_data$comment_score =  sentiment(get_sentences(test_data$comment_cleaned))$sentiment


summary(test_data)

test_data <- test_data %>%
  select(UID,comment_score)

test_data = add_predictions(test_data, model, var = "score", type = NULL)

results_data <- test_data %>%
  select(UID,score)


write.csv(results_data,file = "MyData.csv")
