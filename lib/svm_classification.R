#run visualize_features and get_words before this
#library(caret)

library(reticulate)
np <- import("numpy")
library(e1071)
library(ROCR)
library("mltools")

dec = "" #"dec_" #dec_ or ""
opt = "TR"

args = commandArgs(trailingOnly=TRUE)
var_ind <- as.numeric(args[1])
base <- args[2]
sub <- args[3]

controller_data = read.csv("controller", sep=',')
controller_data <- as.matrix(controller_data)

working_directory <- controller_data[1]
layer <- controller_data[2]
nnodes <- controller_data[3]

is_dec <- args[4]

if(is_dec == 'yes'){
    dec <- 'dec_'
} else{
    dec <- ""
}

asset_directory <- "../assets/"


cat("Variable", var_ind, "\n")


if (opt == "TR") {
  lengths <- c("0s_","1s_", "2s_","4s_","16s_","1600s_"); nlines <- 1295
} else {
  lengths <- c("", "1w_", "2w_", "4w_", "8w_", "16w_", "1s_", "2s_","5s_", "docwise_"); nlines <- 398
}


if (opt == "TR") {
  load(paste(asset_directory, "fullTR.RData", sep="")); Y = fullTR
} else {
  load(paste(asset_directory, "fullSentences_processed.RData", sep="")); Y = fullSentences_processed
}

cat(names(Y)[var_ind],"\n")

run_svm <- function(xtrain, xtest,y,nfold, kernel="linear") {
  set.seed(1)

  dat_train = cbind.data.frame(xtrain,y)
  dat_test = cbind.data.frame(xtest,y)
  names(dat_train) <- c(paste("node",1:nnodes,sep=""),"y")
  names(dat_test) <- c(paste("node",1:nnodes,sep=""),"y")
  preds <- rep(NA, length(y))

  folds = folds(x=dat_train$y,nfolds=nfold,stratified=TRUE,seed=1)

  cat(table(folds), "\n")

  cat(sapply(1:nfold, function (i) sum(dat_train$y[folds==i])), "\n")

  for (i in 1:nfold) {
    train <- dat_train[folds!=i,]
    test <- dat_test[folds==i,]
    m <- svm(factor(y) ~ ., data=train,
                   gamma=1, kernel = kernel, cost = 1)
    preds[folds==i] <- as.logical(predict(m, newdata=test))
  }

  #ROC curve analysis
  pred_object <- prediction(as.numeric(preds), y)
  auc <- slot(performance(pred_object, "auc"),"y.values")[[1]]
  acc <- max(slot(performance(pred_object, "acc"), "y.values")[[1]],na.rm=TRUE)
  f <- max(slot(performance(pred_object, "f"), "y.values")[[1]],na.rm=TRUE)

  print("Confusion matrix:")
  cm <- table(preds, y)
  print(cm)
  print(paste("Accuracy =", acc))
  print(paste("AUC =", auc))
  print(paste("f =", f))

  return(list('preds'=preds, 'cm'=cm, 'acc'=acc, 'auc'=auc, 'f'=f))

}

train_embeddings_filename <- paste(opt, "_", layer, "_context_embeddings.RData", sep="")
load(train_embeddings_filename)

aucs <- list()
cms <- list()
accs <- list()
fs <- list()

for (length in lengths) {

  context <- paste(layer,length,sep="_")

  train_embeddings <- get(context)

  if (dec=="dec_") {
    test_embeddings <- np$load(paste("subject", sub, "_wb_", length, "decoded.npy", sep=""))
  } else {
    test_embeddings <- get(context)
  }

  cat("Context:", context,"\n")

  set.seed(1)
  shuffle_inds <- sample(1:nrow(train_embeddings))
  train_embeddings <- train_embeddings[shuffle_inds,]
  test_embeddings <- test_embeddings[shuffle_inds,]

  var <- Y[shuffle_inds,var_ind]

  #binarize var, whatever it is
  var <- var > 0

  #run svm prediction
  preds <- run_svm(train_embeddings,test_embeddings,var,20)

  aucs[context] <- preds$auc
  accs[context] <- preds$acc
  fs[context] <- preds$f
  cms[context] <- preds$cm

}

varname <- names(Y)[var_ind]

if (dec == "dec_") {
  output_dir <- paste("svm_results_dec/", sep='')
  setwd(output_dir)
} else {
  output_dir <- paste("svm_results/", sep='')
  setwd(output_dir)
}

save(aucs,file=paste("subject", sub, "_wb_", dec, opt, "_", layer ,"_", varname, "_aucs.RData", sep=""))
