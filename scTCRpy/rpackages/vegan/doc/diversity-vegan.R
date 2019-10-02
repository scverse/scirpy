### R code from vignette source 'diversity-vegan.Rnw'

###################################################
### code chunk number 1: diversity-vegan.Rnw:21-26
###################################################
par(mfrow=c(1,1))
options(width=55)
figset <- function() par(mar=c(4,4,1,1)+.1)
options(SweaveHooks = list(fig = figset))
options("prompt" = "> ", "continue" = "  ")


###################################################
### code chunk number 2: diversity-vegan.Rnw:58-60
###################################################
library(vegan)
data(BCI)


###################################################
### code chunk number 3: diversity-vegan.Rnw:78-79
###################################################
H <- diversity(BCI)


###################################################
### code chunk number 4: diversity-vegan.Rnw:86-87
###################################################
J <- H/log(specnumber(BCI))


###################################################
### code chunk number 5: diversity-vegan.Rnw:113-115
###################################################
k <- sample(nrow(BCI), 6)
R <- renyi(BCI[k,])


###################################################
### code chunk number 6: diversity-vegan.Rnw:122-123
###################################################
getOption("SweaveHooks")[["fig"]]()
print(plot(R))


###################################################
### code chunk number 7: diversity-vegan.Rnw:134-135
###################################################
alpha <- fisher.alpha(BCI)


###################################################
### code chunk number 8: diversity-vegan.Rnw:171-172
###################################################
quantile(rowSums(BCI))


###################################################
### code chunk number 9: diversity-vegan.Rnw:175-176
###################################################
Srar <- rarefy(BCI, min(rowSums(BCI)))


###################################################
### code chunk number 10: diversity-vegan.Rnw:184-185
###################################################
S2 <- rarefy(BCI, 2)


###################################################
### code chunk number 11: diversity-vegan.Rnw:189-190
###################################################
all(rank(Srar) == rank(S2))


###################################################
### code chunk number 12: diversity-vegan.Rnw:196-197
###################################################
range(diversity(BCI, "simp") - (S2 -1))


###################################################
### code chunk number 13: diversity-vegan.Rnw:260-264
###################################################
data(dune)
data(dune.taxon)
taxdis <- taxa2dist(dune.taxon, varstep=TRUE)
mod <- taxondive(dune, taxdis)


###################################################
### code chunk number 14: diversity-vegan.Rnw:267-268
###################################################
getOption("SweaveHooks")[["fig"]]()
plot(mod)


###################################################
### code chunk number 15: diversity-vegan.Rnw:294-296
###################################################
tr <- hclust(taxdis, "aver")
mod <- treedive(dune, tr)


###################################################
### code chunk number 16: diversity-vegan.Rnw:318-321
###################################################
k <- sample(nrow(BCI), 1)
fish <- fisherfit(BCI[k,])
fish


###################################################
### code chunk number 17: diversity-vegan.Rnw:324-325
###################################################
getOption("SweaveHooks")[["fig"]]()
plot(fish)


###################################################
### code chunk number 18: diversity-vegan.Rnw:353-354
###################################################
prestondistr(BCI[k,])


###################################################
### code chunk number 19: diversity-vegan.Rnw:385-387
###################################################
rad <- radfit(BCI[k,])
rad


###################################################
### code chunk number 20: diversity-vegan.Rnw:390-391
###################################################
getOption("SweaveHooks")[["fig"]]()
print(radlattice(rad))


###################################################
### code chunk number 21: a
###################################################
sac <- specaccum(BCI)
plot(sac, ci.type="polygon", ci.col="yellow")


###################################################
### code chunk number 22: diversity-vegan.Rnw:461-462
###################################################
getOption("SweaveHooks")[["fig"]]()
sac <- specaccum(BCI)
plot(sac, ci.type="polygon", ci.col="yellow")


###################################################
### code chunk number 23: diversity-vegan.Rnw:490-491
###################################################
ncol(BCI)/mean(specnumber(BCI)) - 1


###################################################
### code chunk number 24: diversity-vegan.Rnw:508-510
###################################################
beta <- vegdist(BCI, binary=TRUE)
mean(beta)


###################################################
### code chunk number 25: diversity-vegan.Rnw:517-518
###################################################
betadiver(help=TRUE)


###################################################
### code chunk number 26: diversity-vegan.Rnw:536-538
###################################################
z <- betadiver(BCI, "z")
quantile(z)


###################################################
### code chunk number 27: diversity-vegan.Rnw:548-553
###################################################
data(dune)
data(dune.env)
z <- betadiver(dune, "z")
mod <- with(dune.env, betadisper(z, Management))
mod


###################################################
### code chunk number 28: diversity-vegan.Rnw:556-557
###################################################
getOption("SweaveHooks")[["fig"]]()
boxplot(mod)


###################################################
### code chunk number 29: diversity-vegan.Rnw:668-669
###################################################
specpool(BCI)


###################################################
### code chunk number 30: diversity-vegan.Rnw:674-676
###################################################
s <- sample(nrow(BCI), 25)
specpool(BCI[s,])


###################################################
### code chunk number 31: diversity-vegan.Rnw:687-688
###################################################
estimateR(BCI[k,])


###################################################
### code chunk number 32: diversity-vegan.Rnw:757-759
###################################################
veiledspec(prestondistr(BCI[k,]))
veiledspec(BCI[k,])


###################################################
### code chunk number 33: diversity-vegan.Rnw:773-774
###################################################
smo <- beals(BCI)


###################################################
### code chunk number 34: a
###################################################
j <- which(colnames(BCI) == "Ceiba.pentandra")
plot(beals(BCI, species=j, include=FALSE), BCI[,j],
     ylab="Occurrence", main="Ceiba pentandra",
     xlab="Probability of occurrence")


###################################################
### code chunk number 35: diversity-vegan.Rnw:787-788
###################################################
getOption("SweaveHooks")[["fig"]]()
j <- which(colnames(BCI) == "Ceiba.pentandra")
plot(beals(BCI, species=j, include=FALSE), BCI[,j],
     ylab="Occurrence", main="Ceiba pentandra",
     xlab="Probability of occurrence")


