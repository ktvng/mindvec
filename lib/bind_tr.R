#library(caret)

library(reticulate)
np <- import("numpy")
#library(e1071)
#library(ROCR)
#library("mltools")

dec = "" #"dec_" #dec_ or ""
opt = "TR"

controller_data = read.csv("controller", sep=',')
controller_data <- as.matrix(controller_data)

working_directory <- controller_data[1]
layer <- controller_data[2]
nnodes <- as.numeric(controller_data[3])

lengths <- c("0s_","1s_", "2s_","4s_","16s_","1600s_")
nlines <- 1295

#setwd(working_directory)

# The following is how the layer 1 embeddings across sentences were bounded into an array for each context,
# and further combined into context_embeddings.RData
ptm <- proc.time()
for (context in lengths) {

 tmp <- matrix(NA,nlines,nnodes)
 for (s in 1:nlines) {
   folder = paste(layer, "_", context, "TRs/", sep="")
   tmp[s,] <- np$load(paste(folder, dec, opt,s,"_",layer, "_", context, "embeddings.npy",sep=""))
 }
 assign(paste(layer, context, sep="_"), tmp)
}
proc.time() - ptm
save(list=(paste(layer,lengths,sep="_")),file=paste(opt, "_", layer, "_context_embeddings.RData", sep=""))

#
