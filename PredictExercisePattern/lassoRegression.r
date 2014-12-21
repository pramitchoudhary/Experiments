## Original Author: cbcb.umd.edu
## Author: Pramit Chodhary


#######
## lasso
library(lars)
library(MASS)
library(ElemStatLearn)
data(prostate)

lassoRegression <- function(){

    covnames <- names(prostate[-(9:10)])
    y <- prostate$lpsa
    x <- prostate[,covnames]
    set.seed(1)
    train.ind <- sample(nrow(prostate), ceiling(nrow(prostate))/2)
    y.test <- prostate$lpsa[-train.ind]
    x.test <- x[-train.ind,]
    y <- prostate$lpsa[train.ind]
    x <- x[train.ind,]
    
    lasso.fit <- lars(as.matrix(x), y, type="lasso", trace=TRUE)
    
    png(file="lasso1.jpg") 
    plot(lasso.fit, breaks=FALSE)
    legend("topleft", covnames, pch=8, lty=1:length(covnames), col=1:length(covnames))
    dev.off()
    
    ## this plots the cross validation curve
    png(file="lasso2.jpg") 
    lasso.cv <- cv.lars(as.matrix(x), y, K=10, type="lasso", trace=TRUE)
    dev.off()
}

lassoRegression()
## Usage system("Rscript lassoRegression.r")
