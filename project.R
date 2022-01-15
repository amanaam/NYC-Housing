#----Libraries----
library(tidyverse)
library(readxl)
library(caret)
library(kernlab)
library(reshape2)
library(pROC)
library(plotly)
library(MLmetrics)
library(xgboost)
options(scipen = 999) # To turn off the scientific notation
#Warning: Boosting Might take extremely long time to train
#---Data Cleaning---------------------------------------------------------------
setwd("C:/Users/Manam/Desktop/Data mining")

data = read.csv("sales.csv")
colnames(data) <- gsub(" ", ".", colnames(data), fixed=TRUE)
str(data)
colnames(data)= tolower(names(data))#fixing column names
View(data)
summary(data)

data[!(apply(data, 1, function(y) any(y == 0))),]#removing entries with 0 as the value

data$sale.price=as.numeric(data$sale.price)
data$gross.square.feet=as.numeric(data$gross.square.feet)
data$land.square.feet=as.numeric(data$land.square.feet)

#Shortlisting the data to ensure its accurate
data = dplyr::filter(data, sale.price>700000, sale.price<10000000, year.built > 1500, residential.units>0,
                     land.square.feet>100, gross.square.feet>100)

#Analysing the neighborhood column to only include neighborhoods with more then 100 sales.
neighborhoods= data%>%
  group_by(neighborhood)%>%
  summarise(count = n())%>%
  arrange(desc(count))
data = merge(data, neighborhoods)
data = filter(data, count>100)

#dummy variables were removed as they were causing problems for the regression.
# data$class1 =ifelse(data$tax.class.at.time.of.sale==1,1,0)
# data$class2=ifelse(data$tax.class.at.time.of.sale==2,1,0)
# data$class3=ifelse(data$tax.class.at.time.of.sale==3,1,0)
# data$class4=ifelse(data$tax.class.at.time.of.sale==4,1,0)

#instead the column was made into factor
data$tax.class.at.time.of.sale = as.factor(data$tax.class.at.time.of.sale)

#selecting the columns to be used for analysis
data=dplyr::select(data,neighborhood,
            block, lot, land.square.feet,
            gross.square.feet, commercial.units, residential.units,
            sale.price, year.built, tax.class.at.time.of.sale)
data = na.omit(data)

#----Data Partition----
set.seed(1)
train.index = sample(row.names(data), dim(data)[1]*0.8)
train.df = data[train.index,]
test.index <- setdiff(row.names(data), train.index)

test.df <- data[test.index,]

#----Models----
#Linear Model
linear_model = lm(sale.price~.,data = train.df)#training model
summary(linear_model)
pred_linear <-predict(linear_model,test.df)#predictions
df_compare_linear <- data.frame(pred_linear,test.df$sale.price, abs(round((test.df$sale.price-pred_linear)*100/test.df$sale.price)))
colnames(df_compare_linear) <- c('pred_price', 'sale.price', "percentageError")

df_compare_linear = df_compare_linear%>%
  filter(pred_price<1000000)
ggplotly(ggplot(data = df_compare_linear)+
           geom_point(aes(sale.price,percentageError)))#plot for error
MAPE(pred_linear,test.df$sale.price)
mean(df_compare_linear$percentageError)#average error

#LASSO Regression
ctrl = trainControl(method="cv",      # simple cross-validation
                    number = 10,      # 10 folds
)
lasso_model = train(sale.price ~ ., data = train.df,
                    method = "lasso",
                    trControl = ctrl
)
summary(lasso_model)
pred_lasso <-predict(lasso_model,test.df)
df_compare_lasso <- data.frame(pred_lasso,test.df$sale.price, abs(round((test.df$sale.price-pred_lasso)*100/test.df$sale.price)))
colnames(df_compare_lasso) <- c('pred_price', 'sale.price', "percentageError")
df_compare_lasso = df_compare_lasso%>%
  filter(pred_price<1000000)
ggplotly(ggplot(data = df_compare_lasso)+
           geom_point(aes(sale.price,percentageError)))
RMSE(pred_lasso,test.df$sale.price)
mean(df_compare_lasso$percentageError)#average error

#SVM
ctrl = trainControl(method="cv",      # simple cross-validation
                    number = 10,      # 10 folds
)
svm_model <- train(
  sale.price ~ ., data = train.df, method = "svmLinear",
  trControl = ctrl,
  preProcess = c("center","scale")
)
summary(svm_model)
pred_svm <-predict(svm_model,test.df)
df_compare_svm <- data.frame(pred_svm,test.df$sale.price, abs(round((test.df$sale.price-pred_svm)*100/test.df$sale.price)))
colnames(df_compare_svm) <- c('pred_price', 'sale.price', "percentageError")
df_compare_svm = df_compare_svm%>%
  filter(pred_price<1000000)
ggplotly(ggplot(data = df_compare_svm)+
           geom_point(aes(sale.price,percentageError)))
RMSE(pred_svm,test.df$sale.price)
mean(df_compare_svm$percentageError)#average error

#XGB boosting Model
boosting_model = train(sale.price~., data = train.df, method = "xgbLinear",
                       objective="reg:squarederror")
pred_boosting <-predict(boosting_model,test.df)

df_compare_boosting <- data.frame(pred_boosting,test.df$sale.price, abs(round((test.df$sale.price-pred_boosting)*100/test.df$sale.price)))
colnames(df_compare_boosting) <- c('pred_price', 'sale.price', "percentageError")
df_compare_boosting = df_compare_boosting%>%
  filter(pred_price<1000000)
ggplotly(ggplot(data = df_compare_boosting)+
           geom_point(aes(sale.price,percentageError)))
RMSE(pred_boosting,test.df$sale.price)
mean(df_compare_boosting$percentageError)

#----Models with Log of Sale Price----
#partitioning 
data$sale.price.log = log(data$sale.price)
train.index.log = sample(row.names(data), dim(data)[1]*0.8)
train.df.log = data[train.index.log,]
test.index.log <- setdiff(row.names(data), train.index.log)
test.df.log <- data[test.index.log,]

#Linear Model
linear_model.log = lm(sale.price.log~.,data = train.df.log)
pred_linear.log <-predict(linear_model.log,test.df.log)
df_compare_linear.log <- data.frame(exp(pred_linear.log),test.df.log$sale.price, abs(round((test.df.log$sale.price-exp(pred_linear.log))*100/test.df.log$sale.price)))
colnames(df_compare_linear.log) <- c('pred_price', 'sale.price', "percentageError")
df_compare_linear.log = df_compare_linear.log%>%
  filter(pred_price<1000000)
ggplotly(ggplot(data = df_compare_linear.log)+
           geom_point(aes(sale.price,percentageError)))
mean(df_compare_linear.log$percentageError)#average error

#LASSO Regression
lasso_model.log = train(sale.price.log ~ ., data = train.df.log,
                    method = "lasso",
                    trControl = ctrl
)
pred_lasso.log <-predict(lasso_model.log,test.df.log)
df_compare_lasso.log <- data.frame(exp(pred_lasso.log), test.df.log$sale.price, abs(round((test.df.log$sale.price-exp(pred_lasso.log))*100/test.df.log$sale.price)))
colnames(df_compare_lasso.log) <- c('pred_price', 'sale.price', "percentageError")
df_compare_lasso.log = df_compare_lasso.log%>%
  filter(pred_price<1000000)
ggplotly(ggplot(data = df_compare_lasso.log)+
           geom_point(aes(sale.price,percentageError)))
RMSE(pred_lasso.log,test.df$sale.price)
mean(df_compare_lasso.log$percentageError)#average error
#SVM
svm_model.log <- train(
  sale.price.log ~ ., data = train.df.log, method = "svmLinear",
  trControl = ctrl,
  preProcess = c("center","scale")
)
pred_svm.log <-predict(svm_model.log,test.df.log)
df_compare_svm.log <- data.frame(exp(pred_svm.log),test.df.log$sale.price, abs(round((test.df.log$sale.price-exp(pred_svm.log))*100/test.df.log$sale.price)))
colnames(df_compare_svm.log) <- c('pred_price', 'sale.price', "percentageError")
df_compare_svm.log = df_compare_svm.log%>%
  filter(pred_price<1000000)
ggplotly(ggplot(data = df_compare_svm.log)+
           geom_point(aes(sale.price,percentageError)))
RMSE(pred_svm.log,test.df.log$sale.price)
mean(df_compare_svm.log$percentageError)

#XGB boosting Model
boosting_model.log = train(sale.price.log~., data = train.df.log, method = "xgbLinear",
                       objective="reg:squarederror")

pred_boosting.log <-predict(boosting_model.log,test.df)

df_compare_boosting.log <- data.frame(exp(pred_boosting.log),test.df.log$sale.price, abs(round((test.df.log$sale.price-exp(pred_boosting.log))*100/test.df.log$sale.price)))
colnames(df_compare_boosting.log) <- c('pred_price', 'sale.price', "percentageError")
df_compare_boosting.log = df_compare_boosting.log%>%
  filter(pred_price<1000000)
ggplotly(ggplot(data = df_compare_boosting.log)+
           geom_point(aes(sale.price,percentageError)))
RMSE(pred_boosting.log,test.df.log$sale.price)
mean(df_compare_boosting.log$percentageError)

#----overall----
results <- data.frame(pred_linear, pred_lasso, pred_svm, pred_boosting,
                      exp(pred_linear.log), exp(pred_lasso.log), exp(pred_svm.log),
                      exp(pred_boosting.log),
                      (pred_linear+pred_lasso+pred_svm+pred_boosting+exp(pred_linear.log)+exp(pred_lasso.log)+exp(pred_svm.log)+exp(pred_boosting.log))/8,
                      abs(round((test.df$sale.price-pred_linear)*100/test.df$sale.price)),
                    abs(round((test.df$sale.price-pred_lasso)*100/test.df$sale.price)),
                    abs(round((test.df$sale.price-pred_svm)*100/test.df$sale.price)),
                    abs(round((test.df$sale.price-pred_boosting)*100/test.df$sale.price)),
                    abs(round((test.df.log$sale.price-exp(pred_linear.log))*100/test.df.log$sale.price)),
                    abs(round((test.df.log$sale.price-exp(pred_lasso.log))*100/test.df.log$sale.price)),
                    abs(round((test.df.log$sale.price-exp(pred_svm.log))*100/test.df.log$sale.price)),
                    abs(round((test.df.log$sale.price-exp(pred_boosting.log))*100/test.df.log$sale.price)),
                    abs(round((test.df$sale.price-(pred_linear+pred_lasso+pred_svm+pred_boosting+pred_boosting.log+pred_lasso.log+pred_linear.log+pred_svm.log)/8)*100/test.df$sale.price)),
                    test.df.log$sale.price)



colnames(results) <- c('LinearModel', 'LassoModel', 'SVMmodel', 'XGBBoosting','LogLinearModel', 'LogLassoModel', 'LogSVMmodel', 'LogXGBBoosting', "AveragePrice",
                       'LinearModelError', 'LassoModelError', 'SVMmodelError', 'XGBBoostingError','LogLinearModelError', 'LogLassoModelError', 'LogSVMmodelError', 'LogXGBBoostingError', "AveragePriceError", "sale.price")
x="ATextIWantToDisplayWithSpaces"
x = gsub('([[:upper:]])', ' \\1', x)
y = strsplit(x, " ")
y=y[[1]][-length(y)][-1]
paste(y, collapse='') 

results = filter(results, LinearModel<1000000, LinearModel>700000, LassoModel<1000000, LassoModel>700000, SVMmodel<1000000, SVMmodel>700000, XGBBoosting<1000000, XGBBoosting>700000)
ggplotly(ggplot(data = results)+
           geom_point(aes(sale.price,LinearModelError))+
           labs(title = "Percentage Errors for Linear Model",
                x= "Sale Prce", y = "Percentage Error")+
           theme_classic())

ggplotly(ggplot(data = results)+
           geom_point(aes(sale.price,LassoModelError))+
           labs(title = "Percentage Errors for LASSO Model",
                x= "Sale Prce", y = "Percentage Error")+
           theme_classic())

ggplotly(ggplot(data = errors)+
           geom_point(aes(sale.price,SVMmodel))+
           labs(title = "Percentage Errors for SVM Model",
                x= "Sale Prce", y = "Percentage Error")+
           theme_classic())
ggplotly(ggplot(data = results)+
           geom_point(aes(sale.price,LogXGBBoostingError))+
           labs(x= "Sale Prce", y = "Percentage Error")+
           theme_classic())

#average error
?aggregate()
View(errorsR)
mean(results$LogXGBBoostingError)
errorsR = results[,11:length(results)-1] 
errorsR.long<-melt(errorsR)
colnames(errorsR.long) = c("Model", "PercentageError")
means<-aggregate(errorsR.long, by=list(errorsR.long$Model), mean)

ggplot(means,aes(x=PercentageError,y=Group.1))+
  geom_bar(stat="identity",position="dodge")+
  xlab("Type of Model")+ylab("Mean Percentage")+
  theme_classic()

view(results)
summary(results)
mean(results$AveragePriceError)
linearcut = cut(results$AveragePriceError, c(0, 20, 50,100), right=FALSE)
total = table(linearcut)
View(cbind(total))

