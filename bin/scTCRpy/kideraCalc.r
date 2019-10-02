#! /usr/local/bioinf/bin/Rscript

#install.packages("Peptides", lib="/home/singlecell/scripts/Tamas/scTCRpy/rpackages", repos="http://cran.us.r-project.org")
library(Peptides, lib="/home/singlecell/scripts/Tamas/scTCRpy/rpackages")

args = commandArgs(trailingOnly=TRUE)
clonotypeFile  <- args[1]
outFile  <- args[2]
#clonotypeFile  <- '/home/szabo/myScratch/RepertoireSeq/reports/clonotype_sequences.txt'
#outFile  <- '/home/szabo/myScratch/RepertoireSeq/reports/clonotype_kidera.txt'

clonotypes <- read.delim(clonotypeFile, header=TRUE, sep='\t', row.names='Subtype')
clonotypes$aTRC <- with(clonotypes, paste(aTRA, aTRB, sep=''))
#clonotypes <- head(clonotypes, n=30)

clonotypes$KiderA <- kideraFactors(clonotypes$aTRA)
clonotypes$KiderB <- kideraFactors(clonotypes$aTRB)
clonotypes$KiderC <- kideraFactors(clonotypes$aTRC)

kiderTypes <- data.frame(clonotypes$Clonotype)
row.names(kiderTypes) <- row.names(clonotypes)
kiderTypes$KiderAs <- sapply(clonotypes$KiderA, paste, collapse='\t')
kiderTypes$KiderBs <- sapply(clonotypes$KiderB, paste, collapse='\t')
kiderTypes$KiderCs <- sapply(clonotypes$KiderC, paste, collapse='\t')

write.table(kiderTypes, file=outFile, col.names=FALSE, quote=FALSE, sep='\t')