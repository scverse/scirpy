#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)
chainpairFile  <- args[1]
outFile  <- args[2]
packageDir  <- args[3]

library(Peptides)

chains <- read.delim(chainpairFile, header=FALSE, sep='\t')
colnames(chains) <- c('chainId', 'TRA', 'TRB')
rownames(chains) <- chains$chainId
chains$TRC <- paste0(chains$TRA, chains$TRB)
head(chains)

chains$KiderA <- kideraFactors(chains$TRA)
chains$KiderB <- kideraFactors(chains$TRB)
chains$KiderC <- kideraFactors(chains$TRC)

kiderTypes <- chains[, c('chainId', 'TRA', 'TRB')]
kiderTypes$KiderAs <- sapply(chains$KiderA, paste, collapse='\t')
kiderTypes$KiderBs <- sapply(chains$KiderB, paste, collapse='\t')
kiderTypes$KiderCs <- sapply(chains$KiderC, paste, collapse='\t')

kiderTypes <- kiderTypes[, -c(2, 3)]

write.table(kiderTypes, file=outFile, row.names=FALSE, col.names=FALSE, quote=FALSE, sep='\t')
