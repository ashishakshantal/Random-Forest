rm(list=ls(all=T))
# setwd("......")
titanic_train <- read.csv("train.csv", header = T, sep = ',')
str(titanic_train)
summary(titanic_train)

#convert attributes to factors
attributes <- names(titanic_train); attributes
to_factors <- c('Survived','Pclass','Sex','Embarked')
to_factors_df <- data.frame(lapply(titanic_train[,to_factors], function(x) as.factor(x)))
titanic_train[,to_factors] <- to_factors_df[,to_factors]

#see summary of attribute 'embarked'
summary(titanic_train$Embarked) #two empty values present. since percentage of rows with empty values is very low
#we can safely remove them
table(titanic_train$Cabin)
titanic_train <- titanic_train[-which(titanic_train$Embarked == ''),]

#should we include cabin? 77% of the attribute values are missing!
titanic_train_new <- subset(titanic_train, select = -c(Cabin))

library(DMwR)
#to check which of the attributes have missing values
apply(titanic_train_new, 2, function(x) which(is.na(x)==TRUE))
summary(titanic_train_new$Age)
length(which(is.na(titanic_train_new$Age==TRUE)))/length(titanic_train_new$Age) #around 20% values missing!

#let's first impute these age values using decision trees

library(rpart)
age_val <- rpart(Age ~ Survived + Pclass + Sex + SibSp + Parch + Embarked + Fare, 
                 data = titanic_train_new[!is.na(titanic_train_new$Age),], method = 'anova')
titanic_train_new$Age[is.na(titanic_train_new$Age)] = predict(age_val, titanic_train_new[is.na(titanic_train_new$Age),])
summary(titanic_train_new)

#using caret, split the data into train-validation as we have test data with us
library(caret)
set.seed(4321)
split_data <- createDataPartition(titanic_train_new$Survived, p = 0.65)
train <- titanic_train_new[split_data$Resample1,]
temp <- titanic_train_new[-split_data$Resample1,]

split_data2 <- createDataPartition(temp$Survived, p = 0.50)
val <- temp[split_data2$Resample1,]
test <- temp[-split_data2$Resample1,]

#implement random forest on the dataset - survival is target variable
# install.packages("randomForest")
library(randomForest)
set.seed(1234)
# names(train)
model_fit <- randomForest(Survived ~ Pclass + Sex + Age + SibSp +
                            Parch + Fare + Embarked,
                          data = train, importance = TRUE, ntree = 200)
#there are other hyper parameters as well - mtry,
#sampsize, nodesize - try using different combinations to see what works best

model_fit$importance
varImpPlot(model_fit)

#confusion matrix for train data
Precision(train$Survived, model_fit$predicted)
Recall(train$Survived, model_fit$predicted)

#confusion matrix for validation data
pred_val <- predict(model_fit, newdata = val, type = 'response')
Precision(val$Survived, pred_val)
Recall(val$Survived, pred_val)

#confusion matrix for test data
pred_test <- predict(model_fit, newdata = test, type = 'response')
Precision(test$Survived, pred_test)
Recall(test$Survived, pred_test)
