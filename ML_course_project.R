# Download and read in data

file.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(file.url, destfile = "pml-training.csv", method = "curl")

file.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(file.url, destfile = "pml-testing.csv", method = "curl")

training <- read.csv("pml-training.csv", na.string = c("", "NA"))
validation <- read.csv("pml-testing.csv", na.string = c("", "NA"))

# Remove variables mostly NA, new window = "yes" rows and label variables
library(dplyr); library(caret)
training  <- filter(training, new_window == "no")

fn <- function(x) sum(is.na(x)) == 0
use_cols <- sapply(training, fn)
use_cols[1:6] <- FALSE
training <- training[, use_cols]
validation <- validation[, use_cols]

# Split training into training and testing data sets
set.seed(1111)
in_train <- createDataPartition(training$classe, p = 0.8, list = FALSE)
testing <- training[-in_train,]
training <- training[in_train,]

# Set trainControl parameters for cross-validation
ctrl <- trainControl(method = "cv")

# Fit a linear discriminant model
library(MASS)
fit.lda <- train(classe ~ . -accel_arm_x,
		     method = "lda",
		     trControl = ctrl,
		     data = training)

pred.lda <- predict(fit.lda, newdata = testing)
LDA <- confusionMatrix(pred.lda, testing$classe)$overall[1]

# Add in principle components analysis
fit.lda.pca <- train(classe ~ . -accel_arm_x -user_name,
			   method = "lda",
			   data = training,
			   preProcess = "pca")

pred.lda.pca <- predict(fit.lda.pca, newdata = testing)
PCA <- confusionMatrix(pred.lda.pca, testing$classe)$overall[1]

# Fit a decision tree model
set.seed(1212)
library(rpart)
fit.cart <- train(classe ~ .,
			method = "rpart",
			trControl = ctrl,
			data = training)

pred.cart <- predict(fit.cart, newdata = testing)
CART <- confusionMatrix(pred.cart, testing$classe)$overall[1]

# Fit a boosted decision tree model
library(gbm)
set.seed(2222)
fit.gbm <- train(classe ~ .,
		     method = "gbm",
		     data = training,
		     trControl = ctrl,
		     verbose = F)

pred.gbm <- predict(fit.gbm, newdata = testing)
GBM <- confusionMatrix(pred.gbm, testing$classe)$overall[1]

# Fit a random forest model
library(randomForest)
set.seed(3333)
fit.rf <- train(classe ~ ., 
		    method = "rf",
		    data = training, 
		    trControl = ctrl,
		    importance = T)
pred.rf <- predict(fit.rf, newdata = testing)
RF <- confusionMatrix(pred.rf, testing$classe)$overall[1]

# Compare models
mod_perf <- data.frame(LDA, CART, GBM, RF)
mod_perf

# Predict on validation data using best model (rf)
pred_val.rf <- predict(fit.rf, newdata = validation)

# File generation function from course instructions
pml_write_files = function(x){
	n = length(x)
	for(i in 1:n){
		filename = paste0("problem_id_",i,".txt")
		write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
	}
}

pml_write_files(pred_val.rf)

# Compare predictions from each fit
pred_val.gbm <- predict(fit.gbm, newdata = validation)
pred_val.lda <- predict(fit.lda, newdata = validation)
pred_val.cart <- predict(fit.cart, newdata = validation)
consensus <- data.frame(pred_val.rf, pred_val.gbm, pred_val.lda, pred_val.cart)
consensus
sum(pred_val.rf == pred_val.gbm)
sum(pred_val.rf == pred_val.lda)
sum(pred_val.rf == pred_val.cart)