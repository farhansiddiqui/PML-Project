#read data
set.seed(1234)
setwd("~/R/PracticalML/Project")
activity <- read.csv(file="pml-training.csv",head=TRUE,sep=",", na.strings=c("NA", "#DIV/0!"))
#explore data set structure
str(activity)
#plot number of NAs
hist(colSums(is.na(activity)), main="# of NAs vs. # Freq of Columns")
#exclude NA columns with more than 95% of NAs
activity <- activity[,colSums(is.na(activity))< 0.95*nrow(activity)]
library(caret)
#partition the data into training and test
inTrain <- createDataPartition(y=activity$classe, p=0.7, list=FALSE)
training <- activity[inTrain,]
testing <- activity[-inTrain,]
#run principal component analysis
pr <-prcomp(training[,-c(1:7,60)], center=T, scale=T)
print(paste("Number of Principal Components : ",length(pr$sdev)))
plot(pr$x,col=training$classe,xlab='PC1',ylab='PC2',main="PC1 vs. PC2; Color = Activity Class")
#create training PC set
trainPC=predict(pr,training[,-c(1:7,60)])
library(randomForest)
#run CV to estimate impact of # of dependent variables on error 
cv <- rfcv(trainPC, training$classe)
cv$error
plot(cv$n.var,cv$error,type="b",xlab="# of predictors",ylab="cv error", main="# of predictors vs. cv error")
#train the model on all variables
modelFit <- randomForest(training$classe~ .,data=trainPC)
modelFit
#create testing PC set
testPC=predict(pr,testing[,-c(1:7,60)])
#evaluate prediction results
confusionMatrix(testing$classe,predict(modelFit,testPC))
#use the model to predict 20 cases
testing <- read.csv(file="pml-testing.csv",head=TRUE,sep=",", na.strings=c("NA", "#DIV/0!"))
testPC=predict(pr,testing[,-c(1:7)])
answers = predict(modelFit,testPC)
answers
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)