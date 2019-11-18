#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Usage example: python summarizeReceptorChainTables.py /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/chainPairs.tsv /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/chainDiv.tsv /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/chainKideras.tsv /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/chainMap.tsv /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/mergedChains.tsv /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/Summa/receptorTable.tsv /home/szabo/myScratch/RepertoireSeq/nextflow_scTCR/allout/Summa/chainTable.tsv

import sys
import numpy as np

def __main__():
    args = sys.argv
    cellPossessions, divtab, kidtab, maptab, chaintab, outFr, outFc = args[1:]
    R, C = {}, {}
    slotpos = {'primary': 0, 'secondary': 1, 'swap': 2}
    with open(cellPossessions) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            cell = line[0]
            i = slotpos[line[1]]
            units = line[2].split(':')
            a, b = units[0].split('.'), units[1].split('.')
            ks = a[0] + ':' + b[0]
            k = a[0] + '.' + a[1] + ':' + b[0] + '.' + b[1]
            for c in (a, b):
                c = c[0] + '.' + c[1]
                if c not in C:
                    C[c] = []
                C[c].append(cell)
            if k not in R:
                R[k] = [ks, 0, 0, 0, [], [], []]
            l = R[k]
            l[i+1] += 1
            l[i+4].append(cell)  
    l0 = ['NA', 0, 0, 0, [], [], []]
    h0 = ['nucLevelId', 'aaLevelId', 'numCellPrimary', 'numCellSecondary', 'numCellSwap', 'cellsPrimary', 'cellsSecondary', 'cellsSwap']
    D1 = {}
    firstline = True
    with open(divtab) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            if firstline:
                firstline = False
                h1 = line[1:]
            else:
                D1[line[0]] = line[1:]
    l1 = ['NA' for x in line[1:]]
    D2 = {}
    h2 = ['KA1', 'KA2', 'KA3', 'KA4', 'KA5', 'KA6', 'KA7', 'KA8', 'KA9', 'KA10', 'KB1', 'KB2', 'KB3', 'KB4', 'KB5', 'KB6', 'KB7', 'KB8', 'KB9', 'KB10', 'KC1', 'KC2', 'KC3', 'KC4', 'KC5', 'KC6', 'KC7', 'KC8', 'KC9', 'KC10']
    with open(kidtab) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            D2[line[0]] = line[1:]
    l2 = ['NA' for x in line[1:]]
    M, H = {}, {}
    with open(maptab) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            ctg = line[0]
            c = line[1].split('.')
            c = c[0] + '.' + c[1]
            M[ctg] = c
            if c not in H:
                H[c] = []
            H[c].append(ctg)
    D3, Q = {}, {}
    h3 = ['is_cell', 'high_confidence', 'length', 'chain', 'v_gene', 'd_gene', 'j_gene', 'c_gene', 'full_length', 'productive', 'reads', 'umis', 'nseq', 'aseq', 'ncdr', 'acdr', 'cdrlen', 'addednuc', 'sample', 'segments']
    with open(chaintab) as f:
        for line in f:
            line = line.split('\r')[0]
            line = line.split('\n')[0]
            line = line.split('\t')
            k = line[0]
            if k in M:
                k = M[k]
                D3[k] = line[2:]
                if k not in Q:
                    Q[k] = []
                Q[k].append(line[12])
    l3 = ['' for x in line[2:]]
    o = '\t'.join(h0+['numReadA', 'meanReadA', 'medianReadA', 'CDRAlen', 'addedAnuc', 'TRAV', 'TRAD', 'TRAJ', 'TRAC', 'numReadB', 'meanReadB', 'medianReadB', 'CDRBlen', 'addedBnuc', 'TRBV', 'TRBD', 'TRBJ', 'TRBC']+h1+h2)+'\n'
    #lines = (set(R.keys())|set(D1.keys())|set(D2.keys()))-set(['10000.1:10000.1']) # Probably an overkill, all biologically sensible receptors should already be in R map.
    lines = set(R.keys())-set(['10000.1:10000.1'])
    for kl in lines:
        if kl in R:
            r0 = R[kl]
        else:
            r0 = l0[:]
        alid = r0[0]
        rz = []
        for e in kl.split(':'):
            if e.split('.')[0] != '10000':
                if e in Q:
                    l = Q[e]
                    li = np.array(l)
                    li = li.astype(np.float)
                    rz += [';'.join(l)+';', str(int(np.mean(li))), str(int(np.median(li)))]
                else:
                    rz += [';', 'NA', 'NA']
                if e in D3:
                    e = D3[e]
                    rz += e[16:18] + e[4:8]
                else:
                    rz += ['None', 'None', 'None', 'None']
            else:
                rz += [';', 'NA', 'NA', 'None', 'None', 'None', 'None']
        r0 = [str(x) for x in r0[:4]] + [';'.join(set(x))+';' for x in r0[4:]] + rz
        if kl in D1:
            r1 = D1[kl]
        else:
            r1 = l1[:]
        if alid in D2:
            r2 = D2[alid]
        else:
            r2 = l2[:]
        o += '\t'.join([kl]+r0+r1+r2)+'\n'
    f = open(outFr, 'w')
    f.write(o)
    f.close()
    o = '\t'.join(['ChainId', 'ChainGroupId', 'NumCells', 'Cells', 'Contigs']+h3)+'\n'
    lines = set(D3.keys())-set(['10000.1'])
    for k in lines:
        km = k.split('.')[0]
        if k in C:
            cells = C[k]
            n = str(len(cells))
            cells = ';'.join(cells)+';'
        else:
            cells, n = ';', '0'
        if k in C:
            cts = ';'.join(H[k])+';'
        else:
            cts = ';'
        o += '\t'.join([k, km, n, cells, cts]+D3[k])+'\n'
    f = open(outFc, 'w')
    f.write(o)
    f.close()
    return

if __name__ == '__main__':
    __main__()