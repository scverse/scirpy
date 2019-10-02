#! /usr/local/bioinf/bin/Rscript

library(vegan, lib="/home/singlecell/scripts/Tamas/scTCRpy/rpackages")
library(DiversitySeq, lib="/home/singlecell/scripts/Tamas/scTCRpy/rpackages")

args = commandArgs(trailingOnly=TRUE)
clonotypeFile  <- args[1]
outFile  <- args[2]

clonotypes <- read.delim(clonotypeFile, header=TRUE, sep='\t', row.names='CLONOTYPE')
clonotypes <- clonotypes[, colSums(clonotypes != 0) > 0]

alphadiv <- aindex(clonotypes, index='Shannon')

write.table(alphadiv, file=outFile, col.names=FALSE, quote=FALSE, sep="\t")