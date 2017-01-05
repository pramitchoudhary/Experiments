# Reference: https://cran.r-project.org/web/packages/ICEbox/ICEbox.pdf
# Data: Boston Housing data

require(ICEbox)
require(randomForest)
require(MASS) # has Boston Housing data, Pima
data(Boston) # Boston Housing data
bostonData = Boston
y = bostonData$medv
X$medv = NULL
# steps to split in train, test and validation is missing

bh_rf = randomForest(bostonData, y)
## Create an 'ice' object for the predictor "age":
bh.ice = ice(object = bh_rf, X = X, y = y, predictor = "age",
             frac_to_build = .1)
# basic plot with PDP( Partial dependency plot)
plot(bh.ice)

## cluster the curves into 2 groups.
clusterICE(bh.ice, nClusters = 2, plot_legend = TRUE)
## cluster the curves into 3 groups, start all at 0.
## TODO:
## 1. Next step to indentify how to use the same to improve prediction
## 2. Add binary feature related to the age(is_gt_60) to see if this improves the model
## in anyway
clusterICE(bh.ice, nClusters = 3, plot_legend = TRUE, center = TRUE)
