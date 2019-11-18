#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Usage example: python chainBasedCellDistanceCalculations.py celldist /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/zd5.h5 /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/chainPairs.tsv minimum 2
#Usage example: python chainBasedCellDistanceCalculations.py cdrseq /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/inToDist1.txt /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/zd5.h5 2
#Usage example: python chainBasedCellDistanceCalculations.py kidera /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/kideras1.txt /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/zd5.h5

import sys, itertools, h5py
import multiprocessing
import numpy as np
from scipy.spatial import distance as disTool
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo

def __main__():
    args = sys.argv
    main_semaphore = args[1]
    if main_semaphore == 'celldist':
        distanceOut, cellChains, disambiguation, numCores = args[2:]
        numCores = int(numCores)
        condensedDistances, chains = readDistances(distanceOut)
        if len(chains) != disTool.num_obs_y(condensedDistances):
            print('Matrix size incompatible with number of observations! Need a condensed distance matrix, not a squareform!')
            return
        chainsOnCells = readMap(cellChains, bottlneck=chains, disambiguation=disambiguation)
        cellDistances, cells = calculateCellDistanceFromChains(condensedDistances, chains, chainsOnCells, disambiguation=disambiguation, numCores=numCores)
        #print(disTool.squareform(cellDistances))
        #print(cells)
        readDistances(distanceOut)
        saveDistMatToHf5(cellDistances, cells, distanceOut, disType='condensedCelldist_')
        readDistances(distanceOut)
        cellDistances = squareFromCondensed(cellDistances, numCores=numCores)
        saveDistMatToHf5(cellDistances, cells, distanceOut, disType='squareCelldist_')
    elif main_semaphore == 'kidera':
        metric = 'euclidean'
        chainpairF, distanceF = args[2:]
        chainNames, chainFeatures = readTab(chainpairF, startcolum=1)
        distanceMatrix = disTool.pdist(chainFeatures, metric=metric)
        #print(disTool.squareform(distanceMatrix))
        saveDistMatToHf5(distanceMatrix, chainNames, distanceF)
    elif main_semaphore == 'cdrseq':
        chainpairF, distanceF, numCores = args[2:]
        numCores = int(numCores)
        chainNames, chainSeqs = readTab(chainpairF)
        distanceMatrix = cdrSeqBasedDistanceCalculation(chainSeqs, numCores=numCores)
        #print(disTool.squareform(distanceMatrix))
        saveDistMatToHf5(distanceMatrix, chainNames, distanceF)
    else:
        print('I do not know this method')
    return

def squareFromCondensed(distances, numCores=2):
    N = len(distances)
    d = int((1+np.sqrt(1+N*8))/2)
    numCores, chunkSize = setChunkSize(numCores, d*d)
    p = multiprocessing.Pool(numCores)
    M = p.map(fillSquareDist, itertools.zip_longest(range(0, d), [], fillvalue=(distances, d)), chunksize=chunkSize)
    p.close()
    p.join()
    return M

def fillSquareDist(params):
    i, paramlist = params
    distances, d = paramlist
    r = []
    for j in range(0, d):
        e = condesedIndexFromSquareIndex(i, j, d)
        if e == None:
            e = 0
        else: 
            e = distances[e]
        r.append(e)
    return r

def calculateCellDistanceFromChains(condensedDistances, chains, chainsOnCells, disambiguation='minimum', numCores=2):
    cellnames = list(chainsOnCells.keys())
    N = len(cellnames)
    L = len(chains)
    posDict = dict(zip(chains, range(L)))
    numCores, chunkSize = setChunkSize(numCores, N*(N-1))
    p = multiprocessing.Pool(numCores)
    M = p.map(checkCellDistance, itertools.zip_longest(itertools.combinations(cellnames, 2), [], fillvalue=(condensedDistances, chainsOnCells, posDict, disambiguation, L)), chunksize=chunkSize)
    p.close()
    p.join()
    M = np.array(M)
    return M, cellnames

def checkCellDistance(arglist):
    cellPair, params = arglist
    condensedDistances, chainsOnCells, posDict, disambiguation, L = params
    c1, c2 = cellPair
    ch1, ch2 = chainsOnCells[c1], chainsOnCells[c2]
    d = []
    foundIdentical = False
    for dc in itertools.product(ch1, ch2):
        dc1, dc2 = dc
        dc1, dc2 = (posDict[dc1], posDict[dc2])
        i = condesedIndexFromSquareIndex(dc1, dc2, L)
        if i == None:
            foundIdentical = True
        else:
            d.append(i)
    d = condensedDistances[d]
    if foundIdentical:
        d = np.append(d, 0)
    if disambiguation == 'maximum':
        d = np.max(d)
    else:
        d = np.min(d)
    return d

def condesedIndexFromSquareIndex(i, j, n):
    if i == j:
        #print('No diagonal elements in condensed matrix')
        return
    else:
        if i < j:
            i, j = j, i
        return int(n*j - j*(j+1)/2 + i - 1 - j)
    return

def cdrSeqBasedDistanceCalculation(chainSeqs, penalties=[-10, -2], numCores=2):
    matrix = MatrixInfo.blosum62
    gap_open, gap_extend = penalties
    N = len(chainSeqs)
    numCores, chunkSize = setChunkSize(numCores, N)
    selfSimilarities = {}
    manager = multiprocessing.Manager()
    selfSimilarities = manager.dict()
    p = manager.Pool(numCores)
    p.map(managedSelfSimFill, itertools.zip_longest(chainSeqs, [], fillvalue=(matrix, selfSimilarities)), chunksize=chunkSize)
    p.close()
    p.join()
    numCores, chunkSize = setChunkSize(numCores, N*(N-1))
    p = manager.Pool(numCores)
    distanceMatrix = p.map(managedDistCalc, itertools.zip_longest(itertools.combinations(chainSeqs, 2), [], fillvalue=(matrix, gap_open, gap_extend, selfSimilarities)), chunksize=chunkSize)
    p.close()
    p.join()
    distanceMatrix = np.array(distanceMatrix)
    maxM, minM = 0.1+np.max(distanceMatrix), np.min(distanceMatrix)
    scaleFactor = maxM - minM
    distanceMatrix += -minM
    distanceMatrix = distanceMatrix/scaleFactor
    distanceMatrix = 1-distanceMatrix
    return distanceMatrix

def managedSelfSimFill(arglist):
    seq, params = arglist
    matrix, selfSimilarities = params
    selfSimilarities[seq[0]] = sum([matrix[(c,c)] for c in seq[1]+seq[2]+seq[1]+seq[2]])
    return

def managedDistCalc(arglist):
    seqPairs, params = arglist
    matrix, gap_open, gap_extend, selfSimilarities = params
    seqPair1, seqPair2 = seqPairs
    n1, a1, b1 = seqPair1
    n2, a2, b2 = seqPair2
    s1, s2 = selfSimilarities[n1], selfSimilarities[n2]
    similarity = distanceOfCDR3Seqs(a1, a2, b1, b2, matrix, gap_open, gap_extend, normFactor=max(s1, s2))
    return similarity

def distanceOfCDR3Seqs(a1, a2, b1, b2, matrix, gap_open, gap_extend, normFactor=1):
    similarity_score = pairwise2.align.globalds(a1, a2, matrix, gap_open, gap_extend, one_alignment_only=True, score_only=True) + pairwise2.align.globalds(b1, b2, matrix, gap_open, gap_extend, one_alignment_only=True, score_only=True)
    similarity_score = similarity_score/normFactor
    return similarity_score

def setChunkSize(numCores, N):
    internalCPUval = multiprocessing.cpu_count()
    if internalCPUval < numCores:
        numCores = internalCPUval
    if numCores > 1:
        chunkSize = int(float(N)/(numCores-1))
    else:
        chunkSize = N
    if chunkSize < 1:
        chunkSize = 1
    return numCores, chunkSize

def readTab(fn, startcolum=0):
    names, M = [], []
    with open(fn) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            M.append(line[startcolum:])
            names.append(line[0])
    return names, M

def readMap(fn, bottlneck=[], disambiguation='minimum'):
    selectors = []
    if disambiguation in ['primary', 'secondary']:
        selectors = [disambiguation]
    else:
        if disambiguation == 'noswap':
            selectors = ['primary', 'secondary']
        else:
            if disambiguation == 'maximum':
                selectors = ['primary', 'secondary']
            else:
                selectors = ['swap', 'primary', 'secondary']
    
    def extractFromLine(line):
        line = line.split('\r')[0]
        line = line.split('\n')[0]
        cell, chainType, chainID = line.split('\t')
        a, b = chainID.split(':')
        a = a.split('.')[0]
        b = b.split('.')[0]
        k = a+':'+b
        return cell, k
    
    def addOnlyIfPresent(fn, bottlneck):
        M = {}
        with open(fn) as f:
            for line in f:
                cell, k = extractFromLine(line)
                if k != '10000:10000':
                    if k in bottlneck:
                        if cell not in M:
                            M[cell] = []
                        M[cell].append(k)
        return M
    
    def addAnyway(fn):
        M = {}
        with open(fn) as f:
            for line in f:
                cell, k = extractFromLine(line)
                if k != '10000:10000':
                    if cell not in M:
                        M[cell] = []
                    M[cell].append(k)
        return M

    if bottlneck == []:
        M = addAnyway(fn)
    else:
        M = addOnlyIfPresent(fn, bottlneck)
    return M

def readDistances(fn, disType=''):
    with h5py.File(fn, 'r') as hf:
        print(hf.keys())
        distances = np.array(hf[disType+'distances'])
        chains = [x.decode(encoding='UTF-8') for x in hf[disType+'names']]
    return distances, chains

def saveDistMatToHf5(M, names, fn, disType=''):
    names = [x.encode(encoding='UTF-8',errors='strict') for x in names]
    with h5py.File(fn, 'a') as hf:
        hf.create_dataset(disType+'distances',  data=M)
        hf.create_dataset(disType+'names',  data=names)
    return

if __name__ == '__main__':
    __main__()