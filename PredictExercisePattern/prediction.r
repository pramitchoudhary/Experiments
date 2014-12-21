## Reference:
## Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements
## Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3JtBv0MRh


# Predict the manner in which the exercises were done
## Read the csv file
filenameTest= "/Users/maverick/Documents/PracticalML/projectDataset/pml-testing.csv"
filenameTrain = "/Users/maverick/Documents/PracticalML/projectDataset/pml-training.csv"
training <- read.csv(filenameTrain, header = TRUE, sep = ",", quote = "\"",
         dec = ".", fill = TRUE, comment.char = "")
x <- subset(training, select =-classe)
y <- training$classe

## Filtering, removing the unwanted variables
myvars <- names(training) %in% c("X", "user_name", "var_yaw_forearm", "stddev_yaw_forearm", "avg_yaw_forearm", "var_pitch_forearm", "stddev_pitch_forearm", "avg_pitch_forearm", "var_roll_forearm", "stddev_roll_forearm", "avg_roll_forearm", "var_accel_forearm", "amplitude_yaw_forearm", "amplitude_pitch_forearm", "amplitude_roll_forearm", "min_roll_forearm", "min_pitch_forearm", "amplitude_yaw_forearm", "min_yaw_forearm", "max_picth_forearm", "max_roll_forearm", "skewness_yaw_forearm")
newdataTraining <- training[!myvars]

## Another way to remove the columns with excessive NAs
newdataTraining<- newdataTraining[,colSums(is.na(newdataTraining)) < 19216]
## to handle empty cells, one needs to convert the same to NA and then use the above technique again
newdataTraining[newdataTraining==""] <- NA

## Feature selection using PCA and applying random forest to train the model, with cross validation
modelFit <- train(newdataTraining$classe ~ ., method="rf", preProcess="pca", trControl = trainControl(method = "cv"), data=newdataTraining)
## Results of the model can be viewed using
modelFit

## The fitted model can now be used to predict on the test data set
pred <- predict(modelFit, test)

## Results can be visualized using plots
qplot(pred, user_name, colour=pred, data=test, main="Prediction on TestData")

## function to generate the result for the test-case
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

