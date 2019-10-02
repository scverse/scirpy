### R code from vignette source 'partitioning.Rnw'

###################################################
### code chunk number 1: partitioning.Rnw:20-26
###################################################
par(mfrow=c(1,1))
figset <- function() par(mar=c(4,4,1,1)+.1)
options(SweaveHooks = list(fig = figset))
library(vegan)
labs <- paste("Table", 1:4)
cls <- c("hotpink", "skyblue", "orange", "limegreen")


###################################################
### code chunk number 2: partitioning.Rnw:39-40
###################################################
getOption("SweaveHooks")[["fig"]]()
showvarparts(2, bg = cls, Xnames=labs)


###################################################
### code chunk number 3: partitioning.Rnw:51-52
###################################################
getOption("SweaveHooks")[["fig"]]()
showvarparts(3, bg = cls, Xnames=labs)


###################################################
### code chunk number 4: partitioning.Rnw:64-65
###################################################
getOption("SweaveHooks")[["fig"]]()
showvarparts(4, bg = cls, Xnames=labs)


