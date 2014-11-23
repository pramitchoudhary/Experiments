#load data
data(iris)
table(iris$Species)
# Using the Caret package to partition the data
inTrain <- createDataPartition( y=iris$Species, p= 0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

#plot
qplot(Petal.Width, Sepal.Width, colour=Species, data=training)

#Classification using Decision Tree
library(rattle)
fancyRpartPlot(modFit$finalModel)

## Bagging [ Bootstrap Aggregation ] ( similar bias and reduced variance, useful for non-linear functions )
## Useful for non-linear functions
## e.g. by 
library(ElemStatLearn)
data(ozone, package="ElemStatLearn")
ozone <- ozone[ order(ozone$ozone), ] # sorting based on a column
## Basic Idea
## Bagged LOESS
ll <- matrix(NA, nrow=10, ncol=155)
for(i in 1:10){
    ss <- sample(1:dim(ozone)[1], replace=T)
    ozone0 <- ozone[ss, ]; ozone0 <- ozone0[ order(ozone$ozone), ]
    loess0 <- loess(temperature ~ ozone, data=ozone0, span=0.2)
    ll[i,] <- predict(loess0, newdata=data.frame(ozone=1:155))
}
plot(ozone$ozone, ozone$temperature, pch=19, cex=0.5)
## Bagging using the caret pkg
predictor = data.frame(ozone=ozone$ozone)
temperature = ozone$temperature
treebag <- bag(predictor, temperature, B=10, bagcontrol= bagControl(fit = ctreeBag$fit, predict= ctreeBag$pred, aggregate = ctreeBag$aggregate ))
plot(ozone$ozone, temperature, col='lightgrey', pch=19)
points(ozone$ozone, predict(treebag$fit[[1]]$fit, predictor), pch=19, col="red")
points(ozone$ozone, predict(treebag, predictor), pch=19, col="blue")

## Random Forest, Caret pkg ( Can be very accurate for wide range of problems )
## Difficult to interpret but very accurate, often used by the Data Science experts
data(iris)
library(ggplot2)
inTrain <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain, ]
testing <- iris[-inTrain, ]
modFit <- train(Species ~ ., data=training, method="rf", prox=TRUE) # Random Forest
getTree(modFit$finalModel, k=2)

irisP <- classCenter(training[, c(3,4)], training$Species, modFit$finalModel$proximity)
irisP <- as.data.frame(irisP); irisP$Species <- rownames(irisP)
p <- qplot(Petal.Width, Petal.Length, col=Species, data=training)
##Predicting new values
pred <- predict(modFit, testing); testing$predRight <- pred==testing$Species
table(pred, testing$Species)
## visualize the prediction
qplot(Petal.Width, Petal.Length, colour=predRight, data=testing, main="newData Predictions")

