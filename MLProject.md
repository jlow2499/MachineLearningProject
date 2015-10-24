---
title: "Predicting Fitbit Exercises Using Decision Trees & Random Forests"
author: "Jeremiah Lowhorn"
date: "Saturday, October 24, 2015"
output: html_document
---

#Executive Summary
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants as they perform barbell lifts correctly and incorrectly 5 different ways.

The classes for the experiment are as follows:
Class A - exactly according to the specification
Class B - throwing the elbows to the front
Class C - lifting the dumbbell only halfway
Class D - lowering the dumbbell only halfway
Class E - throwing the hips to the front

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. Researchers made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).

#Data
The training data set for the project is located at:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data set is available at:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

#Purpose
The purpose of this project is to accurately predict the manner in which the participants did the exercise. This corresponds to the "classe" variable in the training data set. 

#Getting and Cleaning Data
We must first download and clean the data for model building.

###Loading Correct R Packages for Model Building and Cleaning


```r
library(dplyr)
library(stringr)
library(AppliedPredictiveModeling)
library(caret)
library(rattle)
library(rpart.plot)
library(randomForest)
```

###Downloading Data

```r
url_train <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
train <- read.csv(url_train, na.strings=c("NA","#DIV/0!",""), header=TRUE,stringsAsFactors=FALSE)
test <- read.csv(url_test, na.strings=c("NA","#DIV/0!",""), header=TRUE,stringsAsFactors=FALSE)
```

###Cleaning the data, removing variables with near zero variance, and removing any columns with more than 50% of missing data.

We use the nearZeroVar function to remove all variables with a near zero variance for optimum model performance. We will additionally remove all NA values in columns with 50% or higher missing values. The time stamp variable is also cleaned to parse out only the hour in which the participant was measured to examine if a specific time of day had an impact on their performance. Finally we will convert all character vectors to binary classifiers for use in our model building.

```r
###Near Zero Variance Removal###
nzv <- nearZeroVar(train,saveMetrics=TRUE)
train <- train[,nzv$nzv==FALSE]

###Removing NA Values###
nacount <- apply(train,2, function(x) {sum(is.na(x))})
nacount <- as.data.frame(nacount,row.names=names(train))
nacount <- add_rownames(nacount) %>%
  mutate(percentNA = round(nacount/19622 * 100,2)) %>%
  filter(percentNA >= 50)
dropnames <- nacount$rowname

train <- train[,!names(train) %in% dropnames]
train <- train[,!names(train) == "X"]

###Removing the Classe variable so we can bind it to our model matrix later####
classe <- train$classe
train <- train[,!names(train) %in% c("classe")]

###Cleaning the time stamp and only testing which hour the participant was examined###
train$cvtd_timestamp <- gsub("^.*? ","",train$cvtd_timestamp)
time <- as.data.frame(str_split_fixed(train$cvtd_timestamp,":",n=2)) %>%
  rename(Hour = V1)
hour <- as.numeric(as.character(time$Hour))
train <- cbind(train,hour)
train <- train[,names(train) != "cvtd_timestamp"]

###Setting aside the good column names for our data clean function###
goodnames <- names(train)

###Building a model matrix ~ converting character vectors to binary classifiers###
train <- as.data.frame(model.matrix(~0+.,data=train))
train <- cbind(train,classe)
```
###Cross Validation
Here we use 70% of the training data set to train the model and the remaining 30% we will use for model validation. 

```r
set.seed(12345)
Trainer <- createDataPartition(train$classe,p=0.7,list=FALSE)
Training <- train[Trainer,]
Testing <- train[-Trainer,] 
```

#Model Building
###Training With Decision Trees
Decision trees dissect data by asking questions about the data. The questions are derived from variables within the data and explain class variance in order to predict the outcome of the target variable. 

```r
 fit1 <- rpart(classe ~.,data=Training,method="class")
Predictions1 <- predict(fit1,Testing,type="class")
CMTree <- confusionMatrix(Predictions1,Testing$classe)
CMTree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1559  125   14   26    6
##          B   43  806   28   11   66
##          C    1   48  864   59   79
##          D   35   76   45  759   70
##          E   36   84   75  109  861
## 
## Overall Statistics
##                                          
##                Accuracy : 0.824          
##                  95% CI : (0.814, 0.8336)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.7772         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9313   0.7076   0.8421   0.7873   0.7957
## Specificity            0.9594   0.9688   0.9615   0.9541   0.9367
## Pos Pred Value         0.9012   0.8449   0.8221   0.7706   0.7391
## Neg Pred Value         0.9723   0.9325   0.9665   0.9582   0.9532
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2649   0.1370   0.1468   0.1290   0.1463
## Detection Prevalence   0.2940   0.1621   0.1786   0.1674   0.1980
## Balanced Accuracy      0.9453   0.8382   0.9018   0.8707   0.8662
```


```r
fancyRpartPlot(fit1,message=FALSE,warning=FALSE)
```

```
## Error: prp: illegal argument "message"
```

###Training with Random Forests
Random forests construct a multitude of decision trees and average responses across the trees to obtain an estimate of the dependent varaible, thus a radom forest model should be more accurate than a single decision tree. 


```r
fit2 <- randomForest(classe~., data=Training,ntree=2500)
Predictions2 <- predict(fit2,Testing,type="class")
CMRF <- confusionMatrix(Predictions2,Testing$classe)
CMRF 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    2    0    0    0
##          B    1 1137    3    0    0
##          C    0    0 1023    4    0
##          D    0    0    0  960    2
##          E    0    0    0    0 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.998           
##                  95% CI : (0.9964, 0.9989)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9974          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9982   0.9971   0.9959   0.9982
## Specificity            0.9995   0.9992   0.9992   0.9996   1.0000
## Pos Pred Value         0.9988   0.9965   0.9961   0.9979   1.0000
## Neg Pred Value         0.9998   0.9996   0.9994   0.9992   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1932   0.1738   0.1631   0.1835
## Detection Prevalence   0.2846   0.1939   0.1745   0.1635   0.1835
## Balanced Accuracy      0.9995   0.9987   0.9981   0.9977   0.9991
```
###Results
We can ovbserve from the below model statistics that our random forest algorithm is much more accurate at predicting the classe variable than our decision tree. Our in sample error is much lower than that of our decision trees. (83% accuracy vs 99% accuracy) Due to the random forest having a much lower in sample error we will assume that it will produce a very similar out of sample error and will be best for predicting the test results.

###Functions for Test data cleaning & Results Submission
We create a function to clean the test data exactly how we did with the training model, parse out the predictions from the random forest algorithm and write each prediction to a separate file for submission.


```r
prediction_prep <- function(x) {
  x$cvtd_timestamp <- gsub("^.*? ","",x$cvtd_timestamp)
  time <- as.data.frame(str_split_fixed(x$cvtd_timestamp,":",n=2)) %>%
    rename(Hour = V1) 
  hour <- as.numeric(as.character(time$Hour))
  x <- cbind(x,hour) 
  x <- x[,names(x) %in% goodnames]
  x <- as.data.frame(model.matrix(~0+.,data=x))
}  

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

test <- prediction_prep(test)

answers <- predict(fit2,test,type="class")

pml_write_files(answers)
```


#References
Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.

Read more: <http://groupware.les.inf.puc-rio.br/har#sbia_paper_section#ixzz3pVBEiqZv>

