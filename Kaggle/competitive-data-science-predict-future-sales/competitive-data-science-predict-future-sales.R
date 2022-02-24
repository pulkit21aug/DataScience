library(dplyr)
library(forecast)
library(plyr)

#install.packages("forecast",dependencies = TRUE)

df_items <- read.csv("items.csv")
df_items_cat <- read.csv("item_categories.csv")
df_shops <- read.csv("shops.csv")
df_train <- read.csv("sales_train_v2.csv")
df_test <- read.csv("test.csv")


df_train$year <- substr(df_train$date,7,10)

df_train$year <- as.numeric(df_train$year)

df_train$month <- substr(df_train$date,4,5)

df_train$month <- as.numeric(df_train$month)

df_train$date <- as.Date(df_train$date, "%d.%m.%Y")

sapply(df_train,class)

df_test$shop_id = as.numeric(df_test$shop_id)
df_test$item_id = as.numeric(df_test$item_id)


df_train_smry <- df_train %>% group_by(date_block_num, shop_id, item_id, year, month) %>% 
  dplyr::summarize(item_cnt_month = sum(item_cnt_day)) 


getPredictions <- function(storeId, itemId) {

  train_subset <- subset(df_train_smry, df_train_smry$shop_id == storeId & df_train_smry$item_id == itemId)

  if(empty(train_subset)) {
    print("empty subset")
    ts <- ts(data=df_train_smry$item_cnt_month ,start = 2013,end = 2015 , frequency =12)
    model <- auto.arima(ts) 
    pred <- forecast(model,h=1)
    return(pred)
    
  }else {
    ts <- ts(data=train_subset$item_cnt_month ,start = 2013,end = 2015 , frequency =12)
    model <- auto.arima(ts) 
    print("model created")
    pred <- forecast(model,h=1)
    return(pred)
  }
 
}


for(i in 1:nrow(df_test)) {
   shopid <- df_test[i,"shop_id"]
   itemid <- df_test[i,"item_id"]
   print(shopid)
   print(itemid)
  
  predNov <-  getPredictions(shopid,itemid)
  if(is.null(predNov )) {
    df_test[i,"item_cnt_month"] <- 0
  }else {
    df_test[i,"item_cnt_month"] <- predNov$mean[1]
  }
  
  
}

write.csv(df_test,file = "MyData.csv")


