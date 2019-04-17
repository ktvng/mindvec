# Set up parameters
args = commandArgs(trailingOnly=TRUE)
# use actual or decoded?
dec = "wb" # "" or "wb" or "sl5000"
# predict features in the entire context or just the most recent TR?
long = "" # "" or "_long" 

# use sentence or TR
opt = "TR" #sentence or TR
# shuffle labels? (to get a null results)
shuffle <- "" # "" or "_shuffled"
# which variable?
var_ind <- as.numeric(args[1])
cat("Variable", var_ind, "\n")
# which model, which layer?
base <- args[2] #"layer0" or "layer1" or "glove"
# which subject?
sub <- args[3]

controller_data = read.csv("controller", sep=',')
controller_data <- as.matrix(controller_data)

working_directory <- controller_data[1]
layer <- controller_data[2]
nnodes <- controller_data[3]

is_dec <- controller_data[4]

if(is_dec == 'yes'){
    dec <- 'dec_'
} else{
    dec <- ""
}

asset_directory <- "../assets/"

#run visualize_features and get_words before this
#library(caret)
library(reticulate)
np <- import("numpy")
library(e1071)
library(ROCR)
library(PRROC)
library("mltools")

# define possible context lengths
if (opt == "TR") {
  nlines <- 1295
  if (long == "_long") {
    lengths <- c("0s_","1s_", "2s_","4s_")  
  } else {
    lengths <- c("0s_","1s_", "2s_","4s_","16s_","1600s_")  
  }
  
} else {
  lengths <- c("", "1w_", "2w_", "4w_", "8w_", "16w_", "1s_", "2s_","5s_", "docwise_"); nlines <- 398
}

run_svm <- function(xtrain, xtest,y,nfold, kernel) {
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
    m <- svm(factor(y) ~ ., data=train, probability=TRUE,
                   gamma=1, kernel = kernel, cost = 1)
    tmp <- predict(m, newdata=test, probability=TRUE)
    #preds[folds==i] <- tmp
    preds[folds==i] <- attr(tmp,'probabilities')[,'TRUE']
  }
  
  #PR curve analysis
  positive_scores <- preds[y==TRUE]
  negative_scores <- preds[y==FALSE]
  pr<-pr.curve(scores.class0 = positive_scores, scores.class1 = negative_scores)
  pr_auc <- pr$auc.integral
  print(paste("PR AUC:", pr_auc))

  #ROC curve analysis
  roc<-roc.curve(scores.class0 = positive_scores, scores.class1 = negative_scores)
  roc_auc <- roc$auc
  print(paste("ROC AUC:", roc_auc))

  #ROC curve analysis
  #pred_object <- prediction(as.numeric(preds), y)
  #auc <- slot(performance(pred_object, "auc"),"y.values")[[1]]
  #acc <- max(slot(performance(pred_object, "acc"), "y.values")[[1]],na.rm=TRUE)
  #f <- max(slot(performance(pred_object, "f"), "y.values")[[1]],na.rm=TRUE)
  
  #print("Confusion matrix:") 
  #cm <- table(preds, y)
  #print(cm)
  #print(paste("Accuracy =", acc))
  #print(paste("AUC =", auc))
  #print(paste("f =", f))
  
  return(list('preds'=preds, 'pr_auc'=pr_auc, 'roc_auc'=roc_auc))
  #return(list('preds'=preds, 'cm'=cm, 'acc'=acc, 'auc'=auc, 'f'=f))

}

train_embeddings_filename <- paste(opt, "_", layer, "_context_embeddings.RData", sep="")
load(train_embeddings_filename)

pr_aucs <- list()
roc_aucs <- list()
for (length in lengths) {

  if (opt == "TR") {
    if (long == "_long") {
      f = paste(asset_directory, "fullTR_", length, ".RData", sep="")
    } else {
      f = paste(asset_directory, "fullTR.RData", sep="")
    }
    load(f); Y = fullTR
  } else {
    load("fullSentences_processed.RData"); Y = fullSentences_processed
  }
  cat(names(Y)[var_ind],"\n")

  context <- paste(layer,length,sep="_")

  train_embeddings <- get(context)

  if (dec=="dec_") {
    test_embeddings <- np$load(paste("subject", sub, "_wb_", length, "decoded.npy", sep=""))
  } else {
    test_embeddings <- get(context)
  }

  # if (dec!="") {
  #   setwd(path_to_embeddings)
  #   # for GloVe, that data is here: /gpfs/milgram/project/chun/kxt3/official/glove
  #   test_embeddings <- np$load(paste("subject", sub, "_", dec, "_", length, "decoded.npy", sep=""))
  #   setwd(output_path)
  # } else {
  #   test_embeddings <- get(context)
  # }

  cat("Context:", context,"\n")

  set.seed(1)
  shuffle_inds <- sample(1:nrow(train_embeddings))
  train_embeddings <- train_embeddings[shuffle_inds,]
  test_embeddings <- test_embeddings[shuffle_inds,]

  var <- Y[shuffle_inds,var_ind]

  #binarize var, whatever it is
  var <- var > 0  

  # shuffle (for perm test)
  if (shuffle=="_shuffled") {
    var <- sample(var)
  }

  #run svm prediction
  xtrain<-train_embeddings
  xtest<-test_embeddings
  y<-var
  nfold<-20
  kernel<-"linear"
  preds <- run_svm(xtrain,xtest,y,nfold,kernel)

  pr_aucs[context] <- preds$pr_auc
  roc_aucs[context] <- preds$roc_auc

}

varname <- names(Y)[var_ind]

if (dec == "dec_") {
  output_dir <- paste("svm_results_dec/", sep='')
  setwd(output_dir)
} else {
  output_dir <- paste("svm_results/", sep='')
  setwd(output_dir)
}

f <- paste("subject", sub, "_", dec, "_", opt, "_", layer ,"_", varname, long, shuffle, "_aucs.RData", sep="")
save(pr_aucs, roc_aucs,file=f)