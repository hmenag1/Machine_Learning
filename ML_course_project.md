---
title: "Machine Learning Course Project"
author: "Steve Senior"
date: "17 February 2015"
output: html_document
---

## Summary
The task for this assignment was to train an algorithm to predict whether a weight lifting exercise was being performed well or not using data from sensors mounted on the participant and weight. I describe how I have cleaned the data and narrowed down the variables to be used for prediction. I describe my approach to building and testing predictive models. I then fit four different models and assess their likely out-of-sample accuracy using two different approaches.

The model which performed best was a random forest model, which achieved an **estimated out-of-sample accuracy of about 99.8% using both 10-fold cross-validation and a separate sub-sampled test set**. I used this model to successfully predict all the classes of all 20 test samples in the automatically graded component of this course.

## Loading packages
I used the following packages for this project. 

```r
library(dplyr); library(caret); library(MASS); library(plyr)
library(rpart); library(gbm); library(randomForest)
```

## Getting, loading and processing the data
The data for this task is provided by the groupware Human Activity Recognition project. More information on their work can be found [here](http://groupware.les.inf.puc-rio.br/har). This task relates to the weight lifting exercise data. In this study, participants were measured performing an activity well, or making one of four common mistakes. The variable 'classe' captures this information in the training data. A value of 'A' represents a well-performed exercise; 'B' to 'E' represent the errors. The task here is to use sensor data to predict the classe. In other words, classe is the outcome variable.

The code below downloads the data and loads it into R:


```r
file.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(file.url, destfile = "pml-training.csv", method = "curl")
training <- read.csv("pml-training.csv", na.string = c("", "NA"))
dim(training)
```

```
## [1] 19622   160
```

From this we can see that there are 19622 observations of 160 variables. However, a quick inspection shows that a lot of the variables are mostly empty. The code below calculates the number of missing values for each variable. 


```r
na.count <- function(x) sum(is.na(x))
na.vars <- sapply(training, na.count)
sum(na.vars == 0)
```

```
## [1] 60
```

There are 60 variables that contain no missing data and 100 that contain mostly missing data. The background material and the variable names suggest that the variables with the missing data contain summary statistics for each window of data collection. Since the testing data are given as individual time points within a window, we can afford to drop the rows containing summary information and the columns that contain summary variables. Since the goal is to produce an algorithm that can tell whether an unknown user is performing an exercise well or not, I also drop the variables that won't help with this: the 'X' sample number, user name data and the timestamp information. 


```r
training  <- filter(training, new_window == "no") # Drops window summary rows

no.na <- function(x) sum(is.na(x)) == 0
use_cols <- sapply(training, no.na)
use_cols[1:6] <- FALSE
training <- training[, use_cols]
```

We now have a training data set of 19216 observations of 54 variables. I haven't chosen to do any other pre-processing of the data, such as centering or scaling, log transforms or principle components analysis (I did try using PCA to improve the accuracy of the linear discriminant algorithm, but it made things worse).

## Cross-validation design
I use two approaches to estimating the accuracy of the models. The first is to use cross-validation within the training set. By default, the caret package calculates estimates of accuracy by random sampling with replacement (so-called bootstrapping) using 25 repetitions. This may overestimate the accuracy of the model. There are better alternatives that can be accessed using the trainControl function in caret. Here I will use 10-fold cross-validation.

The second approach is to split the training set into a training and testing set. Since we have *a lot* of training data, we can use this approach as well. This is the approach that I report in selecting the best algorithm.


```r
# Split training into training and testing data sets
set.seed(1111)
in_train <- createDataPartition(training$classe, p = 0.8, list = FALSE)
testing <- training[-in_train,]
training <- training[in_train,]

# Set trainControl parameters for cross-validation
ctrl <- trainControl(method = "cv")
```

## Training models
Here I train four models:

* A linear discriminany analysis (lda) model;
* A classification tree model;
* A boosted decision tree model; and
* A random forest model.

### Linear discriminant analysis
Linear discriminant analysis is one of the simplest approaches to classification. This approach tries to find linear boundaries in the variable space that best separate the categories. It is similar to logistic regression, but relies on more assumptions about the data (namely that the distributions of the features within categories are normally distributed). However, it is more naturally suited to classification problems like this one where the outcome variable is multinomial (classe is a factor with five levels).


```r
# Fit a linear discriminant model
fit.lda <- train(classe ~ . -accel_arm_x,
		     method = "lda",
		     trControl = ctrl,
		     data = training)
pred.lda <- predict(fit.lda, newdata = testing)
confusionMatrix(pred.lda, testing$classe)$overall[1]
```

```
## Accuracy 
##    0.705
```

### Decision tree
Another simple model is a decision tree. This approach iteratively looks for the variable and value that best separates the categories. It is naturally suited to classification problems, but may struggle to capture some of the complexity in the data and is prone to overfitting. We can see that this approach performs much worse than the lda model above.


```r
set.seed(1212)
fit.cart <- train(classe ~ .,
			method = "rpart",
			trControl = ctrl,
			data = training)
pred.cart <- predict(fit.cart, newdata = testing)
confusionMatrix(pred.cart, testing$classe)$overall[1]
```

```
## Accuracy 
##   0.4871
```

### Boosted decision trees
Boosting involves fitting lots of decision trees. As the fitting proceeds, data that was mis-classified by previous trees is up-weighted. This has the advantage of gradually eroding the residuals. The trees themselves are typically less complex than a single decision tree approach. While this approach can still overfit the data, it is less prone than a single decision tree model. This approach performs dramatically better than either lda or a single decision tree.


```r
set.seed(2222)
fit.gbm <- train(classe ~ .,
		     method = "gbm",
		     data = training,
		     trControl = ctrl,
		     verbose = F)
pred.gbm <- predict(fit.gbm, newdata = testing)
confusionMatrix(pred.gbm, testing$classe)$overall[1]
```

```
## Accuracy 
##   0.9833
```

### Random forest model
Random forest is essentially a bagged decision tree approach. Like boosting, this involves growing lots of simpler trees. However here trees are grown on random sub-sets of the data (both observations and variables). This approach tends to avoid overfitting and is very popular because it tends to work well "out of the box".


```r
set.seed(3333)
fit.rf <- train(classe ~ ., 
		    method = "rf",
		    data = training, 
		    trControl = ctrl,
		    importance = T)
pred.rf <- predict(fit.rf, newdata = testing)
confusionMatrix(pred.rf, testing$classe)$overall[1]
```

```
## Accuracy 
##   0.9979
```

We can see that the random forest model performs best overall. Based on its performance on the sub-sampled test set, I expect it to have an **out-of-sample error rate of around 0.2%**. Note that this measure of accuracy is similar to that given by 10-fold cross-validation:


```r
fit.rf$results[2,2]
```

```
## [1] 0.998
```

This shows that 10-fold cross-validation can be as good an estimate of accuracy as using a separate test set.

Both the random forest and boosted decision tree models were able to correctly predict all 20 of the test samples used in the automatically graded part of this project. The linear discriminant model achieved 13 out of 20 and the decision tree only got 8.
