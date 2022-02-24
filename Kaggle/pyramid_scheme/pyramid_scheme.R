#calulate net profit
net_profit <- function(cost_price,profit_markup,depth_of_tree,sales_commision) {
  selling_price <-  profit_markup*cost_price
  result <- selling_price - cost_price - (depth_of_tree-1)*sales_commision
  return(result)
}


#simulate a dataset

df <- data.frame(matrix(ncol = 5, nrow = 0))
x <- c("cost_price", "profit_markup", "depth_of_tree","sales_commission","profit")
colnames(df) <- x




for(i in 1:500) {
  cost_price<- 3500
  profit_markup <- sample(2:5,1) 
  depth_of_tree<- sample(2:30,1)
  sales_commission <- 1000
  profit <- as.numeric(net_profit(cost_price,profit_markup,depth_of_tree,sales_commission))
  data <- c(cost_price,profit_markup,depth_of_tree,sales_commission,profit)
  df[i,] <- data
}


write.csv(df,file = 'E:/DataScience/pyramid_scheme.csv')

cor(df)
plot(df$profit~ df$depth_of_tree ,lwd=4,col=2)

reg_model <- lm(df$profit~ df$depth_of_tree,data=df)
summary(reg_model)



