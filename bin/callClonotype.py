#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Usage example: python callClonotype.py /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/mergedCDRs.tsv mappedClonotypes.tsv additionalCellInfo.tsv chainConvergence.tsv chainMap.tsv chainPairs.tsv chainNet.tsv inToDiv.txt inToDist.txt

import sys
from collections import Counter


def __main__():
    (
        none,
        cdrF,
        clonInfo,
        cellInfo,
        convergenceTab,
        chainMap,
        chainPairs,
        chainNet,
        seqsToDiv,
        seqsToDis,
    ) = sys.argv
    distance_method = "minimum"  # How will the distance be calculated - meaning should we include secondary chains and swaps; one of ['minimum', 'maximum', 'noswap', 'primary', 'secondary']
    # Loop through the merged file of CDR sequences and contig IDs for all cells and then
    # Output
    #    a mapping of new clonotype names along with some clonotype information (like sample abundance and convergences),
    #    a mapping of new chain names with some chain features (eg. convergence),
    #    a network of clonotypes and chains that are assigned to more than one clonotype,
    #    additional information on chain pairs of cells (mapping, might refine clonotype categorization)
    #    a list of unique chain pairs as a file input for diversity, Kidera and sequence similarity calculations (two separate files because of convergence on amino acids)
    callClonotypesAndChainPairs(
        cdrF,
        clonInfo,
        cellInfo,
        convergenceTab,
        chainMap,
        chainPairs,
        chainNet,
        seqsToDiv,
        seqsToDis,
        distance_method,
    )
    return


def callClonotypesAndChainPairs(
    cdrF,
    clonInfo,
    cellInfo,
    convergenceTab,
    chainMap,
    chainPairs,
    chainNet,
    seqsToDiv,
    seqsToDis,
    distance_method="minimum",
):
    chainMapping = {}
    cdrMapping = {}
    convergenceTable = {}
    convergenceStats = {}
    clonoFreq = Counter()
    cellChainTable = []
    cellInfoTable = {}
    chainsForDiversity = []
    chainsForDist = []
    sharedChains = {}

    # Read the somewhat preprocessed file and rename chains already, but otherwise only aggregate info for other tasks
    (
        chainMapping,
        cdrMapping,
        convergenceTable,
        clonoFreq,
        cellChainTable,
        cellInfoTable,
        chainsForDiversity,
        chainsForDist,
    ) = renameChain(
        cdrF,
        chainMapping,
        cdrMapping,
        convergenceTable,
        clonoFreq,
        cellChainTable,
        cellInfoTable,
        chainsForDiversity,
        chainsForDist,
        distance_method,
    )

    # Store a list of chain pairs (might not be used later and can then be removed from the script)
    f = open(chainPairs, "w")
    f.write("\n".join(cellChainTable) + "\n")
    f.close()
    cellChainTable = None

    # Just for the record, save the mapping between contigs and chain names (might also become unnecessary later)
    f = open(chainMap, "w")
    f.write("\n".join([x + "\t" + y for x, y in chainMapping.items()]) + "\n")
    f.close()
    chainMapping = None

    # Save the uniqe ntd chain pairs for distance calculations
    f = open(seqsToDiv, "w")
    f.write("\n".join(set(chainsForDiversity)) + "\n")
    f.close()
    chainsForDiversity = None

    # Save the uniqe aa chain pairs for distance calculations
    f = open(seqsToDis, "w")
    f.write("\n".join(set(chainsForDist)) + "\n")
    f.close()
    chainsForDist = None

    # Create human-friendly names for clonotypes, based on overall clonotype abundance
    clonotypeMapping = {"|||": "ct_10000"}
    cloN = 10000
    for m, n in clonoFreq.most_common():
        if m != "|||":
            cloN += 1
            clonotypeMapping[m] = "ct_" + str(cloN)

    # Loop through cells again to count clonotypes per sample, identifiy chains that are shared among clonotypes, assign cell-specific information gathered in previous steps and create the input table for sample-wise diversity calculation
    sharedChains, clonotypePerSample, samples, cellTable = countClonotypesInSamples(
        cellInfoTable, clonotypeMapping, sharedChains
    )

    # Save cell information
    f = open(cellInfo, "w")
    f.write("\n".join(["\t".join(x) for x in cellTable]) + "\n")
    f.close()
    cellTable = None

    # Save the network of clonotypes linked by shared chains
    sharedChains = reformatSharedChains(sharedChains)
    f = open(chainNet, "w")
    f.write("\n".join(["\t".join(x) for x in sharedChains]) + "\n")
    f.close()
    sharedChains = None

    # Loop through the convergenceTable (already ordered by unique CDR3 aa sequences) to calculate nucleotide convergence
    convergenceStats = detailedConvergenceStatistics(
        convergenceTable, cdrMapping, convergenceStats
    )
    f = open(convergenceTab, "w")
    header = "\t".join(convergenceStats.pop("header", ["NA"]))
    f.write(
        "\n".join(
            [header]
            + [
                "\t".join([k] + [str(y) for y in x])
                for k, x in convergenceStats.items()
            ]
        )
        + "\n"
    )
    f.close()

    # Add some convergence statistics to clonotypes
    clonotypeConvergence = clonotypeConvergenceStats(
        clonotypeMapping, convergenceStats, set(cdrMapping["XXXXXXXXXXXX"][1])
    )

    # Assign chain convergence information and sample-wise abundance to clonotypes
    clonotypeInfoTable = reformatClonotypeAbundance(
        clonotypeConvergence, clonotypePerSample, samples
    )
    f = open(clonInfo, "w")
    f.write(
        "\n".join(["\t".join([str(y) for y in x]) for x in clonotypeInfoTable]) + "\n"
    )
    f.close()
    clonotypeInfoTable = None
    return


def renameChain(
    cdrF,
    chainMapping,
    cdrMapping,
    convergenceTable,
    clonoFreq,
    cellChainTable,
    cellInfoTable,
    chainsForDiversity,
    chainsForDist,
    distance_method,
):
    cdrMapping["XXXXXXXXXXXX"] = ["10000", []]
    NaaChain = 10000
    subgroupNames = ["A.1", "B.1", "A.2", "B.2"]
    with open(cdrF) as f:
        for line in f:
            line = line.split("\n")[0]
            line = line.split("\r")[0]
            line = line.split("\t")
            chainLinks = ["10000.1.A.1", "10000.1.B.1", "10000.1.A.2", "10000.1.B.2"]
            cellNames, cellChains = [], []
            for i in range(1, 5):
                aseq, nseq, ctg = line[i], line[i + 4], line[i + 8]
                cellNames.append(ctg.split("_contig_")[0])
                subgroup = subgroupNames[i - 1]
                if aseq == "":
                    aseq = "XXXXXXXXXXXX"
                    line[i] = aseq
                    if ctg != "":
                        chainMapping[ctg] = "10000.1." + subgroup
                else:
                    if aseq not in convergenceTable:
                        NaaChain += 1
                        convergenceTable[aseq] = {}
                        MainChain = NaaChain + 0
                        cdrMapping[aseq] = [str(MainChain), []]
                    else:
                        MainChain = cdrMapping[aseq][0].split(".")[0]
                    cdrMapping[aseq][1].append(cellNames[0])
                    ndict = convergenceTable[aseq]
                    NntChain = len(ndict)
                    if nseq not in ndict:
                        ndict[nseq] = {k: 0 for k in subgroupNames}
                        ndict[nseq]["var"] = str(NntChain)
                    NntChain = ndict[nseq]["var"]
                    ndict[nseq][subgroup] += 1
                    cLink = str(MainChain) + "." + NntChain + "." + subgroup
                    chainMapping[ctg] = cLink
                    chainLinks[i - 1] = cLink
            (
                clonoSign,
                a1aa,
                b1aa,
                a2aa,
                b2aa,
                a1nt,
                b1nt,
                a2nt,
                b2nt,
                a1ctg,
                b1ctg,
                a2ctg,
                b2ctg,
            ) = line
            clonoFreq[clonoSign] += 1
            mainGr = [l.split(".")[0] for l in chainLinks]
            subGr = [".".join(l.split(".")[0:2]) for l in chainLinks]
            div1 = [subGr[0] + ":" + subGr[1], a1aa, b1aa, a1nt, b1nt]
            chainsForDiversity.append("\t".join(div1))
            if distance_method != "secondary":
                dis1 = [mainGr[0] + ":" + mainGr[1], a1aa, b1aa]
                chainsForDist.append("\t".join(dis1))
            cellNames = list(set(cellNames) - set([""]))
            if len(cellNames) == 1:
                cellNames = cellNames[0]
                cdrMapping["XXXXXXXXXXXX"][1].append(cellNames)
                cellChains.append(
                    [cellNames, "primary", chainLinks[0] + ":" + chainLinks[1]]
                )
                cellChains.append(
                    [cellNames, "secondary", chainLinks[2] + ":" + chainLinks[3]]
                )
                if chainLinks[2] + ":" + chainLinks[3] != "10000.1.A.2:10000.1.B.2":
                    cellChains.append(
                        [cellNames, "swap", chainLinks[0] + ":" + chainLinks[3]]
                    )
                    cellChains.append(
                        [cellNames, "swap", chainLinks[2] + ":" + chainLinks[1]]
                    )
                    div2 = [subGr[2] + ":" + subGr[3], a2aa, b2aa, a2nt, b2nt]
                    chainsForDiversity.append("\t".join(div2))
                    if distance_method in ["minimum", "maximum", "noswap", "secondary"]:
                        dis2 = [mainGr[2] + ":" + mainGr[3], a2aa, b2aa]
                        chainsForDist.append("\t".join(dis2))
                        if distance_method in ["minimum", "maximum"]:
                            if a2aa != "XXXXXXXXXXXX" or b1aa != "XXXXXXXXXXXX":
                                dis = [mainGr[2] + ":" + mainGr[1], a2aa, b1aa]
                                chainsForDist.append("\t".join(dis))
                            if b2aa != "XXXXXXXXXXXX" or a1aa != "XXXXXXXXXXXX":
                                dis = [mainGr[0] + ":" + mainGr[3], a1aa, b2aa]
                                chainsForDist.append("\t".join(dis))
                cellInfoTable[cellNames] = [
                    clonoSign,
                    cellNames[: cellNames.rindex("_") + 1],
                    chainLinks,
                ]
            cellChainTable += ["\t".join(x) for x in cellChains]
    return (
        chainMapping,
        cdrMapping,
        convergenceTable,
        clonoFreq,
        cellChainTable,
        cellInfoTable,
        chainsForDiversity,
        chainsForDist,
    )


def countClonotypesInSamples(cellInfoTable, clonotypeMapping, sharedChains):
    clonotypePerSample = {x: Counter() for x in clonotypeMapping.values()}
    cellTable = []
    samples = []
    for cell, cellinfo in cellInfoTable.items():
        signiture, sample, chains = cellinfo
        signiture = clonotypeMapping[signiture]
        chainmains = [x.split(".")[0] for x in chains]
        for chain in chainmains:
            if chain not in sharedChains:
                sharedChains[chain] = []
            sharedChains[chain].append(signiture)
        if sample not in samples:
            samples.append(sample)
        clonotypePerSample[signiture][sample] += 1
        cellTable.append(
            [cell, signiture, ":".join(chains), ":".join(chainmains)]
            + chains
            + chainmains
            + [
                chains[0] + ":" + chains[1],
                chains[2] + ":" + chains[3],
                chains[0] + ":" + chains[2],
                chains[1] + ":" + chains[3],
            ]
            + [
                chainmains[0] + ":" + chainmains[1],
                chainmains[2] + ":" + chainmains[3],
                chainmains[0] + ":" + chainmains[3],
                chainmains[2] + ":" + chainmains[1],
                chainmains[0] + ":" + chainmains[2],
                chainmains[1] + ":" + chainmains[3],
            ]
        )
    return sharedChains, clonotypePerSample, samples, cellTable


def detailedConvergenceStatistics(
    convergenceTable,
    cdrMapping,
    convergenceStats=[],
    chainNames=["A.1", "B.1", "A.2", "B.2"],
    lineprefix="",
    includeHeaders=True,
):
    headrow = [
        "aaSeq",
        "cells",
        "chainID",
        "ntdVars",
        "sumOccurrence",
        "positionFidelity",
        "dominanceFidelity",
        "alphaConv",
        "betaConv",
        "domConv",
        "secConv",
        "alphaOccurrence",
        "betaOccurrence",
        "dominantOccurrence",
        "secondaryOccurrence",
        "domApOccurrence",
        "secApOccurrence",
        "domBpOccurrence",
        "secBpOccurrence",
        "alphaRatio",
        "betaRatio",
        "domRatio",
        "secRatio",
    ]
    if isinstance(convergenceStats, list):
        listForm = True
    else:
        if isinstance(convergenceStats, dict):
            listForm = False
        else:
            return
    convergenceTable.pop("", None)
    for aa, nt in convergenceTable.items():
        a1, b1, a2, b2 = chainNames
        ntdVars = len(nt)
        sumOccurrence = 0
        alphaOccurrence = 0
        betaOccurrence = 0
        dominantOccurrence = 0
        secondaryOccurrence = 0
        domApOccurrence = 0
        secApOccurrence = 0
        domBpOccurrence = 0
        secBpOccurrence = 0
        alphaConv = 0
        betaConv = 0
        domConv = 0
        secConv = 0
        for nchain, groupings in nt.items():
            sumOccurrence += groupings[a1]
            sumOccurrence += groupings[a2]
            sumOccurrence += groupings[b1]
            sumOccurrence += groupings[b2]
            alphaOccurrence += groupings[a1]
            alphaOccurrence += groupings[a2]
            betaOccurrence += groupings[b1]
            betaOccurrence += groupings[b2]
            dominantOccurrence += groupings[a1]
            dominantOccurrence += groupings[b1]
            secondaryOccurrence += groupings[a2]
            secondaryOccurrence += groupings[b2]
            domApOccurrence += groupings[a1]
            secApOccurrence += groupings[a2]
            domBpOccurrence += groupings[b1]
            secBpOccurrence += groupings[b2]
            if groupings[a1] > 0 or groupings[a2] > 0:
                alphaConv += 1
            if groupings[b1] > 0 or groupings[b2] > 0:
                betaConv += 1
            if groupings[a1] > 0 or groupings[b1] > 0:
                domConv += 1
            if groupings[a2] > 0 or groupings[b2] > 0:
                secConv += 1
        alphaRatio = float(alphaOccurrence) / sumOccurrence
        betaRatio = float(betaOccurrence) / sumOccurrence
        domRatio = float(dominantOccurrence) / sumOccurrence
        secRatio = float(secondaryOccurrence) / sumOccurrence
        positionFidelity = max(alphaRatio, betaRatio)
        dominanceFidelity = max(domRatio, secRatio)
        chaincode, cellist = cdrMapping[aa]
        cellist = set(cellist) - set([""])
        cellist = ";".join(cellist)
        if listForm:
            convergenceStats.append(
                [
                    lineprefix,
                    aa,
                    cellist,
                    chaincode,
                    ntdVars,
                    sumOccurrence,
                    positionFidelity,
                    dominanceFidelity,
                    alphaConv,
                    betaConv,
                    domConv,
                    secConv,
                    alphaOccurrence,
                    betaOccurrence,
                    dominantOccurrence,
                    secondaryOccurrence,
                    domApOccurrence,
                    secApOccurrence,
                    domBpOccurrence,
                    secBpOccurrence,
                    alphaRatio,
                    betaRatio,
                    domRatio,
                    secRatio,
                ]
            )
        else:
            convergenceStats[aa] = [
                cellist,
                chaincode,
                ntdVars,
                sumOccurrence,
                positionFidelity,
                dominanceFidelity,
                alphaConv,
                betaConv,
                domConv,
                secConv,
                alphaOccurrence,
                betaOccurrence,
                dominantOccurrence,
                secondaryOccurrence,
                domApOccurrence,
                secApOccurrence,
                domBpOccurrence,
                secBpOccurrence,
                alphaRatio,
                betaRatio,
                domRatio,
                secRatio,
            ]
    if includeHeaders:
        if listForm:
            convergenceStats = [headrow] + convergenceStats
        else:
            convergenceStats["header"] = headrow
    return convergenceStats


def clonotypeConvergenceStats(clonotypeMapping, convergenceStats, allCells):
    convergenceStats[""] = [";".join(allCells), "10000", 0, 0, 0, 0]
    ctConv = {}
    for sign, ct in clonotypeMapping.items():
        convStat = [sign]
        cdras = sign.split("|")
        convStatPart = []
        cells = set(allCells)
        for i in range(0, len(cdras)):
            aa = cdras[i]
            cStats = convergenceStats[aa][:6]
            cells = cells & set(cStats[0].split(";"))
            convStatPart += cStats[1:]
        ctConv[ct] = convStat + [";".join(cells)] + convStatPart
    return ctConv


def reformatClonotypeAbundance(clonotypeConvergence, clonotypePerSample, samples):
    clonotypePerSample.pop("ct_10000", None)
    M = []
    for ct, counts in clonotypePerSample.items():
        r = clonotypeConvergence[ct]
        r += [counts[x] for x in samples]
        M.append([ct] + r)
    M.sort(key=lambda x: int(x[0].split("ct_")[1]))
    M = [
        [
            "Clonotype",
            "Signiture",
            "Cells",
            "cdrIdA1",
            "nucvarA1",
            "numchainA1",
            "positionA1",
            "dominanceA1",
            "cdrIdB1",
            "nucvarB1",
            "numchainB1",
            "positionB1",
            "dominanceB1",
            "cdrIdA2",
            "nucvarA2",
            "numchainA2",
            "positionA2",
            "dominanceA2",
            "cdrIdB2",
            "nucvarB2",
            "numchainB2",
            "positionB2",
            "dominanceB2",
        ]
        + samples
    ] + M
    return M


def reformatSharedChains(sharedChains):
    sharedChains.pop("10000", None)
    M = []
    for chain, cts in sharedChains.items():
        if len(cts) > 1:
            for ct in cts:
                M.append([ct, chain])
    return M


if __name__ == "__main__":
    __main__()
