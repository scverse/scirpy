### R code from vignette source 'decision-vegan.Rnw'

###################################################
### code chunk number 1: decision-vegan.Rnw:21-26
###################################################
figset <- function() par(mar=c(4,4,1,1)+.1)
options(SweaveHooks = list(fig = figset))
options("prompt" = "> ", "continue" = "  ")
options(width = 55)
require(vegan)


###################################################
### code chunk number 2: decision-vegan.Rnw:84-85 (eval = FALSE)
###################################################
## options(mc.cores = 2)


###################################################
### code chunk number 3: decision-vegan.Rnw:125-136 (eval = FALSE)
###################################################
## ## start up and define meandist()
## library(vegan)
## data(sipoo)
## meandist <-
##     function(x) mean(vegdist(x, "bray"))
## library(parallel)
## clus <- makeCluster(4)
## clusterEvalQ(clus, library(vegan))
## mbc1 <- oecosimu(dune, meandist, "r2dtable",
##                  parallel = clus)
## stopCluster(clus)


###################################################
### code chunk number 4: decision-vegan.Rnw:240-251
###################################################
getOption("SweaveHooks")[["fig"]]()
data(sipoo)
mod <- nestedtemp(sipoo)
plot(mod, "i")
x <- mod$c["Falcsubb"]
y <- 1 - mod$r["Svartholm"]
points(x,y, pch=16, cex=1.5)
abline(x+y, -1, lty=2)
f <- function(x, p) (1-(1-x)^p)^(1/p)
cross <- function(x, a, p) f(x,p) - a + x
r <- uniroot(cross, c(0,1), a = x+y, p = mod$p)$root
arrows(x,y, r, f(r, mod$p), lwd=4)


###################################################
### code chunk number 5: decision-vegan.Rnw:549-553
###################################################
library(vegan)
data(varespec)
data(varechem)
orig <- cca(varespec ~ Al + K, varechem)


###################################################
### code chunk number 6: a
###################################################
plot(orig, dis=c("lc","bp"))


###################################################
### code chunk number 7: decision-vegan.Rnw:562-563
###################################################
getOption("SweaveHooks")[["fig"]]()
plot(orig, dis=c("lc","bp"))


###################################################
### code chunk number 8: decision-vegan.Rnw:572-574
###################################################
i <- sample(nrow(varespec))
shuff <- cca(varespec[i,] ~ Al + K, varechem)


###################################################
### code chunk number 9: decision-vegan.Rnw:577-578
###################################################
getOption("SweaveHooks")[["fig"]]()
plot(shuff, dis=c("lc","bp"))


###################################################
### code chunk number 10: a
###################################################
plot(procrustes(scores(orig, dis="lc"),
                scores(shuff, dis="lc")))


###################################################
### code chunk number 11: decision-vegan.Rnw:591-592
###################################################
getOption("SweaveHooks")[["fig"]]()
plot(procrustes(scores(orig, dis="lc"),
                scores(shuff, dis="lc")))


###################################################
### code chunk number 12: decision-vegan.Rnw:600-603
###################################################
tmp1 <- rda(varespec ~ Al + K, varechem)
i <- sample(nrow(varespec)) # Different shuffling
tmp2 <- rda(varespec[i,] ~ Al + K, varechem)


###################################################
### code chunk number 13: decision-vegan.Rnw:606-608
###################################################
getOption("SweaveHooks")[["fig"]]()
plot(procrustes(scores(tmp1, dis="lc"),
                scores(tmp2, dis="lc")))


###################################################
### code chunk number 14: decision-vegan.Rnw:625-627
###################################################
orig
shuff


###################################################
### code chunk number 15: decision-vegan.Rnw:632-633
###################################################
getOption("SweaveHooks")[["fig"]]()
plot(procrustes(orig, shuff))


###################################################
### code chunk number 16: decision-vegan.Rnw:646-651
###################################################
tmp1 <- rda(varespec ~ ., varechem)
tmp2 <- rda(varespec[i,] ~ ., varechem)
proc <- procrustes(scores(tmp1, dis="lc", choi=1:14),
                   scores(tmp2, dis="lc", choi=1:14))
max(residuals(proc))


###################################################
### code chunk number 17: decision-vegan.Rnw:663-666
###################################################
data(dune)
data(dune.env)
orig <- cca(dune ~ Moisture, dune.env)


###################################################
### code chunk number 18: decision-vegan.Rnw:671-672
###################################################
getOption("SweaveHooks")[["fig"]]()
plot(orig, dis="lc")


###################################################
### code chunk number 19: a
###################################################
plot(orig, display="wa", type="points")
ordispider(orig, col="red")
text(orig, dis="cn", col="blue")


###################################################
### code chunk number 20: decision-vegan.Rnw:696-697
###################################################
getOption("SweaveHooks")[["fig"]]()
plot(orig, display="wa", type="points")
ordispider(orig, col="red")
text(orig, dis="cn", col="blue")


