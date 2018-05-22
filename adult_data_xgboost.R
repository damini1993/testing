######importing the dataset
getwd()
library(data.table)
library(mlr)
#install.packages("mlr")

##Setting the columns for text file
setcol <- c("age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "target")
adult_data <- read.table("adult.data.txt", header = F, sep = ",", col.names = setcol, na.strings = c(" ?"), stringsAsFactors = F)
dim(adult_data)

##setting the target variable to 0 and 1 type
adult_data$target_new=ifelse(adult_data$target==' <=50K',0,1)
adult_data=adult_data[,-15]


####splitting the data into train and test
indexes = sample(1:nrow(adult_data), size=0.3*nrow(adult_data))

# Split data
test = adult_data[indexes,]
dim(test)  # 6 11
train = adult_data[-indexes,]
dim(train) # 26 11

##Convert dataframe to data table
setDT(train) 
setDT(test)

##checking the missing values
colSums(is.na(train))
sapply(train, function(x) sum(is.na(x))/length(x))*100
sapply(test, function(x) sum(is.na(x))/length(x))*100

##removing extra character from the target variable
#library(stringr)
#test [,target := substr(target,start = 1,stop = nchar(target)-1)]
#char_col <- colnames(train)[ sapply (test,is.character)]
#for(i in char_col) set(train,j=i,value = str_trim(train[[i]],side = "left"))
#for(i in char_col) set(test,j=i,value = str_trim(test[[i]],side = "left"))

#set all missing value as "Missing" 
train[is.na(train)] <- "Missing" 
test[is.na(test)] <- "Missing"

###Important things for xgboost model
##Convert the categorical variables into numeric using one hot encoding
##For classification, if the dependent variable belongs to class factor, convert it to numeric

###converting categorical variables to numeric using dummy variables
labels <- train$target_new 
ts_label <- test$target_new
new_tr <- model.matrix(~.+0,data = train[,-c("target_new"),with=F]) 
new_ts <- model.matrix(~.+0,data = test[,-c("target_new"),with=F])

#convert factor to numeric 
#labels <- as.numeric(labels)-1
#ts_label <- as.numeric(ts_label)-1


#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
xgb.DMatrix.save(dtrain, 'xgb.DMatrix.data')
dtrain <- xgb.DMatrix('xgb.DMatrix.data')


dtest <- xgb.DMatrix(data = new_ts,label=ts_label)
xgb.DMatrix.save(dtest, 'xgb.DMatrix.data')
dtest <- xgb.DMatrix('xgb.DMatrix.data')

##building the xgboost model
library(xgboost)
dim(dtrain)
dim(dtest)

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)


xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 79, watchlist = list(val=dtest,train=dtrain), print_every_n = 10, early_stopping_rounds = 10, maximize = F , eval_metric = "error")

#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#confusion matrix
library(caret)
confusionMatrix (xgbpred, ts_label)
