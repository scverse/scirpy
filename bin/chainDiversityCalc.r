#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)
chainFile  <- args[1]
outFile  <- args[2]
packageDir  <- args[3]

library(stringr)
library(vegan)
library(DiversitySeq, lib=packageDir)

nucBet <- c('G', 'A', 'T', 'C')
aaBet <- c('G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S', 'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T')
clonotypes <- read.delim(chainFile, header=FALSE, sep='\t', stringsAsFactors=FALSE)
colnames(clonotypes) <- c('ChainID', 'aTRA', 'aTRB', 'nTRA', 'nTRB')
rownames(clonotypes) <- clonotypes$ChainID
clonotypes$nTRC <- paste0(clonotypes$nTRA, clonotypes$nTRB)
clonotypes$aTRC <- paste0(clonotypes$aTRA, clonotypes$aTRB)

for (chain in c('nTRA', 'nTRB', 'nTRC')){
    for (alp in nucBet){
        colN <- paste(alp, chain, sep='_')
        clonotypes[, colN] <- str_count(clonotypes[, chain], alp)
    }
}
for (chain in c('aTRA', 'aTRB', 'aTRC')){
    for (alp in aaBet){
        colN <- paste(alp, chain, sep='_')
        clonotypes[, colN] <- str_count(clonotypes[, chain], alp)
    }
}

head(clonotypes)
divTab <- data.frame(clonotypes$ChainID)
rownames(divTab) <- rownames(clonotypes)

addNucDiv <- function(inTab, divTab, outname){
  traTab <- subset(inTab, rowSums(inTab)>0)
  traTab <- as.matrix(traTab)
  traTab <- t(traTab)
  traTab <- as.data.frame(traTab)
  traDiv <- aindex(traTab, index='Shannon')
  names(traDiv) <- c(outname)
  traDiv <- as.data.frame(traDiv)
  divTab <- merge(divTab, traDiv, all=TRUE, by='row.names')
  row.names(divTab) <- divTab$Row.names
  divTab$Row.names <- NULL
  return(divTab)
}

inTab <- subset(clonotypes, select=G_nTRA:C_nTRA)
divTab <- addNucDiv(inTab, divTab, 'ShannonNucA')
inTab <- subset(clonotypes, select=G_nTRB:C_nTRB)
divTab <- addNucDiv(inTab, divTab, 'ShannonNucB')
inTab <- subset(clonotypes, select=G_nTRC:C_nTRC)
divTab <- addNucDiv(inTab, divTab, 'ShannonNucC')

inTab <- subset(clonotypes, select=G_aTRA:T_aTRA)
divTab <- addNucDiv(inTab, divTab, 'ShannonProtA')
inTab <- subset(clonotypes, select=G_aTRB:T_aTRB)
divTab <- addNucDiv(inTab, divTab, 'ShannonProtB')
inTab <- subset(clonotypes, select=G_aTRC:T_aTRC)
divTab <- addNucDiv(inTab, divTab, 'ShannonProtC')

colnames(divTab)[[1]] <- 'ChainID'

write.table(divTab, file=outFile, row.names=FALSE, col.names=TRUE, quote=FALSE, sep="\t")
