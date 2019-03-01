#opt
args = commandArgs(trailingOnly=TRUE)
controller <- "controller"

controller_data = read.csv(controller, sep=',')
controller_data <- as.matrix(controller_data)

working_directory <- controller_data[1]
layer <- paste(controller_data[2], "_", sep="")
nnodes <- controller_data[3]
is_dec <- controller_data[4]

if(is_dec == 'yes'){
    dec <- "dec_"
} else {
    dec <- ""
}
#dec <- ""#"dec_"
opt <- "TR_" #"TR_", or "sentence_"

#type
chars <- c("draco","filch","harry","herm","hooch","minerva","neville","peeves","ron","wood")
verbs <- c("be","hear","know","see","tell")
#speech <- c("speak_sticky","speak")
speech <- c("speak")
#motion <- c("fly_sticky","manipulate_sticky","move_sticky","collidePhys_sticky","fly","manipulate","move")
motion <- c("fly","manipulate","move")
#emotion <- c("annoyed","commanding","dislike","fear","like","nervousness","questioning","wonder","annoyed_sticky",
#	"commanding_sticky","cynical_sticky","dislike_sticky","fear_sticky","hurtMental_sticky","hurtPhys_sticky",
#	"like_sticky","nervousness_sticky","pleading_sticky","praising_sticky","pride_sticky","questioning_sticky",
#	"relief_sticky","wonder_sticky")
emotion <- c("annoyed_sticky","commanding_sticky","cynical_sticky","dislike_sticky","fear_sticky","hurtMental_sticky","hurtPhys_sticky",
"like_sticky","nervousness_sticky","pleading_sticky","praising_sticky","pride_sticky","questioning_sticky",
"relief_sticky","wonder_sticky")
visual <- c("word_length","var_WL","sentence_length")
pos <- c(",",".",":","CC","CD","DT","IN","JJ","MD","NN","NNP","NNS","POS","PRP","PRP$","RB","RP","TO","UH","VB",
	"VBD","VBG","VBN","VBP","VBZ","WDT","WP","WRB")
dependency_role <- c("ADV","AMOD","CC","COORD","DEP","IOBJ","NMOD","OBJ","P","PMOD",
	"PRD","PRN","PRT","ROOT","SBJ","VC","VMOD")

for (sub in c("1","2","3","4","5","6","7","8")) {
	for (type in c("chars","verbs","speech","motion","emotion","pos","dependency_role")) {
        if(dec == "dec_"){
            setwd("svm_results_dec")
        } else {
            setwd("svm_results")
        }

        all_aucs <- NULL
		for (var in get(type)) {
            if (file.exists(paste("subject", sub, "_wb_", dec, opt, layer, var,"_aucs.RData",sep=""))) load(paste("subject", sub, "_wb_", dec, opt, layer, var,"_aucs.RData",sep=""))
			all_aucs <- cbind(all_aucs, unlist(aucs))
		}

		setwd("..")
		if(dec == "dec_"){
			setwd("aucs_dec")
		} else {
			setwd("aucs")
		}

		avg_aucs <- rowMeans(all_aucs)

		save(all_aucs,file=paste(dec, type,"_aucs.RData",sep=""))

		library(RColorBrewer)
		n <- length(get(type))
		qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
		col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
		palette(col_vector)

		setwd("..")
		if(dec == "dec_"){
			setwd("plots_dec")
		} else {
			setwd("plots")
		}

		pdf(paste("subject", sub, "_wb_", dec, opt, layer, type,"_aucs.pdf",sep=""),width=14,height=6)
		par(xpd = T, mar = par()$mar + c(0,0,0,7))
		matplot(all_aucs,type=c('b'),col=1:length(names(avg_aucs)),xaxt='n',bty='L',pch=16, ylim = c(0.48,1.0),
			xlab='Context before segment', ylab='ROC AUC',main=paste("LSTM context effect for each features: ",type,sep=""))
		axis(1,at=1:length(names(avg_aucs)),labels=substr(names(avg_aucs),8,20))
		lines(avg_aucs, type=c('b'),lwd=2,pch=17)
		#legend('bottom',legend=get(type),col=1:length(names(avg_aucs)),pch=2, horiz=TRUE)
		legend('right',pch=16,legend=get(type),col=1:length(names(avg_aucs)), cex=1,horiz=FALSE,
			inset = c(-0.1,0),xpd=TRUE)
		par(xpd=F)
		abline(h=0.5, lty=2)
		dev.off()
	    setwd("..")
    }
}
