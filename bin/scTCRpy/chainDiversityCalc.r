#! /usr/local/bioinf/bin/Rscript

#install.packages("vegan", lib="/home/singlecell/scripts/Tamas/scTCRpy/rpackages", repos="http://cran.us.r-project.org")
#install.packages("/home/singlecell/scripts/Tamas/scTCRpy/rpackages/DiversitySeq.tar.gz", lib="/home/singlecell/scripts/Tamas/scTCRpy/rpackages", repos=NULL)

library(vegan, lib="/home/singlecell/scripts/Tamas/scTCRpy/rpackages")
library(DiversitySeq, lib="/home/singlecell/scripts/Tamas/scTCRpy/rpackages")
library(stringr)

args = commandArgs(trailingOnly=TRUE)
clonotypeFile  <- args[1]
outFile  <- args[2]
#clonotypeFile  <- '/home/szabo/myScratch/RepertoireSeq/reports/clonotype_sequences.txt'
#outFile  <- '/home/szabo/myScratch/RepertoireSeq/reports/clonotype_diverse.txt'

nucBet <- c('G', 'A', 'T', 'C')
aaBet <- c('G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S', 'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T')
clonotypes <- read.delim(clonotypeFile, header=TRUE, sep='\t', row.names='Subtype')
clonotypes$nTRC <- with(clonotypes, paste(nTRA, nTRB, sep=''))
clonotypes$aTRC <- with(clonotypes, paste(aTRA, aTRB, sep=''))
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
diverTypes <- data.frame(clonotypes$Clonotype)
row.names(diverTypes) <- row.names(clonotypes)
divTab <- subset(clonotypes, select=Clonotype)

#data(salivaSimData)
#alphadiv <- aindex(simCounts, index='Shannon')
#head(alphadiv)

#Assess Shannon diversity scores for each nucleotide sequence of the TRA and TRB chains

traTab <- subset(clonotypes, select=G_nTRA:C_nTRA)
traTab <- subset(traTab, rowSums(traTab)>0)
traTab <- as.matrix(traTab)
#traTab <- head(traTab, n=100)
traTab <- t(traTab)
traTab <- as.data.frame(traTab)
traDiv <- aindex(traTab, index='Shannon')
names(traDiv) <- c('Shannon_nucA')
traDiv <- as.data.frame(traDiv)
divTab <- merge(divTab, traDiv, all=TRUE, by='row.names')
row.names(divTab) <- divTab$Row.names
divTab$Row.names <- NULL

trbTab <- subset(clonotypes, select=G_nTRB:C_nTRB)
trbTab <- subset(trbTab, rowSums(trbTab)>0)
trbTab <- as.matrix(trbTab)
#trbTab <- head(trbTab, n=100)
trbTab <- t(trbTab)
trbTab <- as.data.frame(trbTab)
trbDiv <- aindex(trbTab, index='Shannon')
names(trbDiv) <- c('Shannon_nucB')
trbDiv <- as.data.frame(trbDiv)
divTab <- merge(divTab, trbDiv, all=TRUE, by='row.names')
row.names(divTab) <- divTab$Row.names
divTab$Row.names <- NULL

trcTab <- subset(clonotypes, select=G_nTRC:C_nTRC)
trcTab <- subset(trcTab, rowSums(trcTab)>0)
trcTab <- as.matrix(trcTab)
#trcTab <- head(trcTab, n=100)
trcTab <- t(trcTab)
trcTab <- as.data.frame(trcTab)
trcDiv <- aindex(trcTab, index='Shannon')
names(trcDiv) <- c('Shannon_nucC')
trcDiv <- as.data.frame(trcDiv)
divTab <- merge(divTab, trcDiv, all=TRUE, by='row.names')
row.names(divTab) <- divTab$Row.names
divTab$Row.names <- NULL

#Assess Shannon diversity scores for each amino acid sequence of the TRA and TRB chains

traTab <- subset(clonotypes, select=G_aTRA:T_aTRA)
traTab <- subset(traTab, rowSums(traTab)>0)
traTab <- as.matrix(traTab)
#traTab <- head(traTab, n=100)
traTab <- t(traTab)
traTab <- as.data.frame(traTab)
traDiv <- aindex(traTab, index='Shannon')
names(traDiv) <- c('Shannon_aaA')
traDiv <- as.data.frame(traDiv)
divTab <- merge(divTab, traDiv, all=TRUE, by='row.names')
row.names(divTab) <- divTab$Row.names
divTab$Row.names <- NULL

trbTab <- subset(clonotypes, select=G_aTRB:T_aTRB)
trbTab <- subset(trbTab, rowSums(trbTab)>0)
trbTab <- as.matrix(trbTab)
#trbTab <- head(trbTab, n=100)
trbTab <- t(trbTab)
trbTab <- as.data.frame(trbTab)
trbDiv <- aindex(trbTab, index='Shannon')
names(trbDiv) <- c('Shannon_aaB')
trbDiv <- as.data.frame(trbDiv)
divTab <- merge(divTab, trbDiv, all=TRUE, by='row.names')
row.names(divTab) <- divTab$Row.names
divTab$Row.names <- NULL

trcTab <- subset(clonotypes, select=G_aTRC:T_aTRC)
trcTab <- subset(trcTab, rowSums(trcTab)>0)
trcTab <- as.matrix(trcTab)
#trcTab <- head(trcTab, n=100)
trcTab <- t(trcTab)
trcTab <- as.data.frame(trcTab)
trcDiv <- aindex(trcTab, index='Shannon')
names(trcDiv) <- c('Shannon_aaC')
trcDiv <- as.data.frame(trcDiv)
divTab <- merge(divTab, trcDiv, all=TRUE, by='row.names')
row.names(divTab) <- divTab$Row.names
divTab$Row.names <- NULL

write.table(divTab, file=outFile, col.names=FALSE, quote=FALSE, sep="\t")