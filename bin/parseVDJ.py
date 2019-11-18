#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, json


def __main__():
    none, prefix, contigF, consenseF, cellF, chainF, cdrF = sys.argv
    processResultsOf10xRepSeq(prefix, contigF, consenseF, cellF, chainF, cdrF)
    return


def processResultsOf10xRepSeq(prefix, contigF, consenseF, cellF, chainF, cdrF):
    sampleCells = {}
    sampleChains = {}
    cellTable = []
    cdrTable = []
    chainTable = []
    contigTable = {}
    consensusTable = {}

    ## Most information is readily avalable in the contig file, we just have to read it in memory. Only cells listed in this table will be used for clonotype assignment (filtered vs. all).
    contigHeader, contigTable, sampleCells = readContigTable(
        prefix, contigF, contigTable, sampleCells
    )
    contigHeader["contigID"] = 0

    ## Some minor things, like added nucleotides can be calculated from the json file only, so we parse that as well (would be perfect if we had to parse the flitered contigs only, but there is json only for unfiltered contigs and at some point we agreed to use the filtered datasets)
    consensusTable, sampleChains = readConsensusTable(
        prefix, consenseF, consensusTable, sampleChains
    )
    # Go through the contig table to find non-consensus(possibly borken) contigs and also to keep as much info for downstream steps as possible
    chainHeader, chainTable, consensusTable = harmonizeContgisAndConsensus(
        prefix, sampleCells, sampleChains, contigHeader, contigTable, consensusTable
    )

    # Real work starts with redefining clonotypes (introducing dominant and secondary TCR chains to overcome issues caused by double-TCR cells) and summing up info available for each cell
    cellHeader = [
        "barcode",
        "sample",
        "clonotype",
        "chainPairing",
        "fragmentChains",
        "dominantAreads",
        "dominantAcdrlen",
        "dominantAaddnuc",
        "dominantAcontig",
        "dominantAfragment",
        "dominantAgeneV",
        "dominantAgeneD",
        "dominantAgeneJ",
        "dominantBreads",
        "dominantBcdrlen",
        "dominantBaddnuc",
        "dominantBcontig",
        "dominantBfragment",
        "dominantBgeneV",
        "dominantBgeneD",
        "dominantBgeneJ",
        "secondaryAreads",
        "secondaryAcdrlen",
        "secondaryAaddnuc",
        "secondaryAcontig",
        "secondaryAfragment",
        "secondaryAgeneV",
        "secondaryAgeneD",
        "secondaryAgeneJ",
        "secondaryBreads",
        "secondaryBcdrlen",
        "secondaryBaddnuc",
        "secondaryBcontig",
        "secondaryBfragment",
        "secondaryBgeneV",
        "secondaryBgeneD",
        "secondaryBgeneJ",
    ]
    for cell in sampleCells.keys():
        cellContigs = sampleCells[cell]
        cellchains = {
            "TRA": ["", "", 0, 0, 0, "", False, "NONE", "NONE", "NONE"],
            "TRB": ["", "", 0, 0, 0, "", False, "NONE", "NONE", "NONE"],
            "_TRA": ["", "", 0, 0, 0, "", False, "NONE", "NONE", "NONE"],
            "_TRB": ["", "", 0, 0, 0, "", False, "NONE", "NONE", "NONE"],
        }
        cellchains = summarizeChainInfoForCell(
            cellContigs,
            consensusTable,
            contigTable,
            contigHeader,
            sampleChains,
            cellchains,
        )
        cloneID = "|".join([cellchains[x][0] for x in ["TRA", "TRB", "_TRA", "_TRB"]])
        pairingCat = decideChainPairingCategory(cellchains)
        cellinfo = [cell, prefix, cloneID, pairingCat, "0"]
        cra, crn, crc = [], [], []
        numfrag = 0
        for tc in ["TRA", "TRB", "_TRA", "_TRB"]:
            tci = cellchains[tc]
            tci[2], tci[3], tci[4] = str(tci[2]), str(tci[3]), str(tci[4])
            if tci[6]:
                numfrag += 1
                tci[6] = "Fragment"
            else:
                tci[6] = "Fulllength"
            cellinfo += tci[2:]
            cra.append(tci[0])
            crn.append(tci[1])
            crc.append(tci[5])
        cellinfo[4] = str(numfrag)
        cellTable.append("\t".join(cellinfo))
        cdrTable.append([cellinfo[2]] + cra + crn + crc)

    # Save the information we gathered
    cellTable = ["\t".join(cellHeader)] + cellTable
    f = open(cellF, "w")
    f.write("\n".join(cellTable) + "\n")
    f.close()
    chainTable = ["\t".join(x) for x in chainTable]
    f = open(chainF, "w")
    f.write("\n".join(chainTable) + "\n")
    f.close()
    cdrTable = ["\t".join(x) for x in cdrTable]
    f = open(cdrF, "w")
    f.write("\n".join(cdrTable) + "\n")
    f.close()
    return


def readContigTable(prefix, contigF, contigTable, sampleCells):
    with open(contigF) as f:
        firstline = True
        for line in f:
            line = line.split("\r")[0]
            line = line.split("\n")[0]
            line = line.split(",")
            if firstline:
                firstline = False
                contigHeader = dict(zip(line, range(0, len(line))))
            else:
                key = prefix + line[2]
                cellbar = prefix + line[contigHeader["barcode"]]
                if cellbar not in sampleCells:
                    sampleCells[cellbar] = []
                sampleCells[cellbar].append(key)
                readcol = contigHeader["reads"]
                reads = line[readcol]
                try:
                    reads = int(reads)
                except:
                    reads = 0
                line[readcol] = reads
                cdr3 = line[contigHeader["cdr3"]]
                if cdr3 == "None":
                    cdr3 = ""
                cdr3 = cdr3.replace("*", "X")
                line[contigHeader["cdr3"]] = cdr3
                cdr3_nt = line[contigHeader["cdr3_nt"]]
                if cdr3_nt == "None":
                    cdr3_nt = ""
                line[contigHeader["cdr3_nt"]] = cdr3_nt
                contigTable[key] = line
    return contigHeader, contigTable, sampleCells


def readConsensusTable(prefix, consenseF, consensusTable, sampleChains):
    with open(consenseF) as f:
        data = json.load(f)
        for clone in data:
            contid, clonotype, seq, aas, info = (
                prefix + clone["contig_name"],
                prefix + clone["clonotype"],
                clone["sequence"],
                clone["aa_sequence"],
                clone["info"],
            )
            cdrAA, cdrNuc, cdrStart, cdrStop = (
                clone["cdr3"],
                clone["cdr3_seq"],
                clone["cdr3_start"],
                clone["cdr3_stop"],
            )
            highConf, mayProd, beenFilt = (
                clone["high_confidence"],
                clone["productive"],
                clone["filtered"],
            )
            inserted, segments, segline, chain = collectSegments(
                contid, clone["annotations"], cdrStart, cdrStop
            )
            consensusTable[contid] = {
                "nonproductive": False,
                "clonotype": clonotype,
                "nseq": seq,
                "aseq": aas,
                "ncdr": cdrNuc,
                "acdr": cdrAA,
                "conf": highConf,
                "prod": mayProd,
                "filt": beenFilt,
                "chain": chain,
                "segline": segline,
                "segments": segments,
                "cdrlen": len(cdrAA),
                "addednuc": inserted,
            }
            for k, v in info.items():
                if k not in ["cells", "cell_contigs"]:
                    consensusTable[contid][k] = v
            consensusTable[contid]["cells"] = [prefix + x for x in info["cells"]]
            consensusTable[contid]["cell_contigs"] = [
                prefix + x for x in info["cell_contigs"]
            ]
            for schain in info["cell_contigs"]:
                schain = prefix + schain
                sampleChains[schain] = contid
    return consensusTable, sampleChains


def harmonizeContgisAndConsensus(
    prefix, sampleCells, sampleChains, contigHeader, contigTable, consensusTable
):
    chainTable = []
    fragment_n = 1000
    chainHeader = list(contigHeader.items())
    chainHeader.sort(key=lambda x: x[1])
    chainHeader = [x[0] for x in chainHeader] + [
        "nseq",
        "aseq",
        "ncdr",
        "acdr",
        "cdrlen",
        "addednuc",
        "sample",
        "segments",
    ]
    chainHeader = (
        chainHeader[:3] + chainHeader[4:13] + chainHeader[15:17] + chainHeader[19:]
    )
    for key, ctg in contigTable.items():
        ctg = [key] + ctg
        seginfo = []
        if key in sampleChains:
            supplements = consensusTable[sampleChains[key]]
            spp = [
                supplements["nseq"],
                supplements["aseq"],
                supplements["ncdr"],
                supplements["acdr"],
                supplements["cdrlen"],
                supplements["addednuc"],
            ]
            for e in supplements["segments"]:
                seginfo.append("/".join([str(x) for x in e]))
        else:
            spp = ["", "", "", "", 0, 0]
            fragment_n += 1
            fkey = prefix + ctg[1] + "_fragment_" + str(fragment_n)
            sampleCells[prefix + ctg[1]].append(key)
            sampleChains[key] = fkey
            consensusTable[fkey] = {
                "nonproductive": True,
                "ncdr": "",
                "acdr": "",
                "chain": ctg[6],
                "segline": [ctg[7].upper(), ctg[8].upper(), ctg[9].upper()],
                "cdrlen": 0,
                "addednuc": 0,
            }
        chainrow = ctg[:3] + ctg[4:13] + ctg[15:17] + spp + [prefix, ";".join(seginfo)]
        chainrow = [str(x) for x in chainrow]
        chainTable.append(chainrow)
    return chainHeader, chainTable, consensusTable


def collectSegments(barcode, annotations, cdrStart, cdrStop):
    segments, segline = [], {"V": "NONE", "D": "NONE", "J": "NONE"}
    for anne in annotations:
        segments.append(
            [
                anne["feature"]["gene_name"],
                anne["contig_match_start"],
                anne["contig_match_end"],
            ]
        )
    chain = anne["feature"]["chain"]
    segments.sort(key=lambda x: x[2])
    previ, inserted = 0, 0
    for s in segments:
        difi = s[1] - previ
        if previ >= cdrStart:
            if s[1] <= cdrStop:
                inserted += difi
        previ = s[2]
        if previ >= cdrStart:
            if s[1] <= cdrStop:
                gc = s[0][3]
                segline[gc] = s[0]
    segline = [segline[k] for k in ["V", "D", "J"]]
    return inserted, segments, segline, chain


def summarizeChainInfoForCell(
    cellContigs, consensusTable, contigTable, contigHeader, sampleChains, cellchains
):
    for ctg in cellContigs:
        rowdata = contigTable[ctg]
        cdr3 = rowdata[contigHeader["cdr3"]]
        cdr3_nt = rowdata[contigHeader["cdr3_nt"]]
        chain = rowdata[contigHeader["chain"]]
        reads = rowdata[contigHeader["reads"]]
        arowdata = consensusTable[sampleChains[ctg]]
        cdrlen = arowdata["cdrlen"]
        addednuc = arowdata["addednuc"]
        frg = arowdata["nonproductive"]
        cellinf = [cdr3, cdr3_nt, reads, cdrlen, addednuc, ctg, frg] + arowdata[
            "segline"
        ]
        if chain in ["TRA", "TRB"]:
            if cellchains[chain][0] == "":
                cellchains[chain] = cellinf
            else:
                if reads > cellchains[chain][2]:
                    if cdr3 == "":
                        if (
                            cellchains["_" + chain] == ""
                            and reads > cellchains["_" + chain][2]
                        ):
                            cellchains["_" + chain] = cellinf
                    else:
                        cellchains["_" + chain] = cellchains[chain][:]
                        cellchains[chain] = cellinf
                else:
                    cellchains["_" + chain] = cellinf
    return cellchains


def decideChainPairingCategory(cellchains):
    pairingCat = "Missing"  # ----
    if cellchains["TRA"][0] == "":
        if cellchains["TRB"][0] == "":
            pass
        else:
            if cellchains["_TRB"][0] == "":
                pairingCat = "OrphanBeta"  # -B--
            else:
                pairingCat = "DoubleOrphanBeta"  # -B-B
    else:
        if cellchains["TRB"][0] == "":
            if cellchains["_TRA"][0] == "":
                pairingCat = "OrphanAlpha"  # A---
            else:
                pairingCat = "DoubleOrphanAlpha"  # A-A-
        else:
            if cellchains["_TRB"][0] == "":
                if cellchains["_TRA"][0] == "":
                    pairingCat = "SinglePair"  # AB--
                else:
                    pairingCat = "DoubleAlpha"  # ABA-
            else:
                if cellchains["_TRA"][0] == "":
                    pairingCat = "DoubleBeta"  # AB-B
                else:
                    pairingCat = "FullDoublet"  # ABAB
    return pairingCat


# This function might be a duplicate of one in the clonotype caller script (import later?)
def calculateConvergenceStats(convergeTable, prefix):
    convergeStat = []
    for aa, nt in convergeTable.items():
        if aa != "":
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
                sumOccurrence += groupings["TRA"]
                sumOccurrence += groupings["_TRA"]
                sumOccurrence += groupings["TRB"]
                sumOccurrence += groupings["_TRB"]
                alphaOccurrence += groupings["TRA"]
                alphaOccurrence += groupings["_TRA"]
                betaOccurrence += groupings["TRB"]
                betaOccurrence += groupings["_TRB"]
                dominantOccurrence += groupings["TRA"]
                dominantOccurrence += groupings["TRB"]
                secondaryOccurrence += groupings["_TRA"]
                secondaryOccurrence += groupings["_TRB"]
                domApOccurrence += groupings["TRA"]
                secApOccurrence += groupings["_TRA"]
                domBpOccurrence += groupings["TRB"]
                secBpOccurrence += groupings["_TRB"]
                if groupings["TRA"] > 0 or groupings["_TRA"] > 0:
                    alphaConv += 1
                if groupings["TRB"] > 0 or groupings["_TRB"] > 0:
                    betaConv += 1
                if groupings["TRA"] > 0 or groupings["TRB"] > 0:
                    domConv += 1
                if groupings["_TRA"] > 0 or groupings["_TRB"] > 0:
                    secConv += 1
            alphaRatio = float(alphaOccurrence) / sumOccurrence
            betaRatio = float(betaOccurrence) / sumOccurrence
            domRatio = float(dominantOccurrence) / sumOccurrence
            secRatio = float(secondaryOccurrence) / sumOccurrence
            positionFidelity = max(alphaRatio, betaRatio)
            dominanceFidelity = max(domRatio, secRatio)
            convergeStat.append(
                [
                    aa,
                    prefix,
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
    return convergeStat


if __name__ == "__main__":
    __main__()
